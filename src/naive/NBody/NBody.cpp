////////////////////////////////////////////////////////////////////////////////////////////////////
//
// OpenCL Kernel taken from AMDAPPSDK
// Program Structure from APPLE QJulia
// Remixed by Xiang
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <GL/glew.h>
#include <GL/glx.h>
#include <GL/freeglut.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>

////////////////////////////////////////////////////////////////////////////////

#define USE_GL_ATTACHMENTS              (0)  // enable OpenGL attachments for Compute results
#define DEBUG_INFO                      (0)
#define COMPUTE_KERNEL_FILENAME         ("NBody_Kernels.cl")
#define COMPUTE_KERNEL_MATMUL_NAME      ("nbody_sim")
#define SEPARATOR                       ("----------------------------------------------------------------------\n")

////////////////////////////////////////////////////////////////////////////////

static GLuint                            VaoID;
static GLuint                            VboPosID[2];
static GLuint                            VboPosLoc[2];
static GLuint                            UniformCurBufIdxLocation;
static GLuint                            VertexShaderID;
static GLuint                            FragShaderID;
static GLuint                            GLProgramID;

////////////////////////////////////////////////////////////////////////////////

FILE *fp;

////////////////////////////////////////////////////////////////////////////////


const char *VertexShaderSource = 
"#version 430\n"

"layout(location=0) in vec4 pos0;\n"
"layout(location=1) in vec4 pos1;\n"

"out vec4 ex_Color;\n"

"uniform int curBufIdx;\n"

"void main(void)\n"
"{\n"
"   if ( curBufIdx == 0 )\n"
"     gl_Position = vec4((pos0.x - 25) / 40, (pos0.y - 30) / 40 , 1.0, 1.0);\n"
"   else\n"
"     gl_Position = vec4((pos1.x - 25) / 40, (pos1.y - 30) / 40, 1.0, 1.0);\n"
"   ex_Color = vec4(1.0, 1.0, 1.0, 1.0);\n"
"}\n";

const char *FragShaderSource = 
"#version 430\n"

"in vec4 ex_Color;\n"
"out vec4 out_Color;\n"

"void main(void)\n"
"{\n"
"   out_Color = ex_Color;\n"
"}\n";

////////////////////////////////////////////////////////////////////////////////

static cl_context                       ComputeContext;
static cl_command_queue                 ComputeCommands;
static cl_kernel                        ComputeKernel;
static cl_program                       ComputeProgram;
static cl_device_id                     ComputeDeviceId;
static cl_device_type                   ComputeDeviceType;
static cl_mem                           ComputePosBuffer[2];
static cl_mem                           ComputeVelBuffer[2];
static size_t                           MaxWorkGroupSize;
static int                              WorkGroupSize[1];
static int                              WorkGroupItems = 32;
static int                              CurrentBuffer = 0;

////////////////////////////////////////////////////////////////////////////////

static int Animated                     = 0;
static int Update                       = 1;

static int WindowWidth                  = 512;
static int WindowHeight                 = 512;

static float *DataInput                 = NULL;

static int DataParticleCount            = 1024;
static int DataBodyCount                = 1024;
static float delT                       = 0.005f;
static float espSqr                     = 500.0f;

static int GroupSize                    = 128;

////////////////////////////////////////////////////////////////////////////////

static double TimeElapsed               = 0;
static int FrameCount                   = 0;
static int NDRangeCount                 = 0;
static uint ReportStatsInterval         = 30;

static float ShadowTextColor[4]         = { 0.0f, 0.0f, 0.0f, 1.0f };
static float HighlightTextColor[4]      = { 0.9f, 0.9f, 0.9f, 1.0f };
static uint TextOffset[2]               = { 25, 25 };

static uint ShowStats                   = 1;
static char StatsString[512]            = "\0";
static uint ShowInfo                    = 1;
static char InfoString[512]             = "\0";

// static float VertexPos[4][2]            = { { -1.0f, -1.0f },
// { +1.0f, -1.0f },
// { +1.0f, +1.0f },
// { -1.0f, +1.0f } };

////////////////////////////////////////////////////////////////////////////////
static long
GetCurrentTime()
{
	struct timeval tv;

	gettimeofday(&tv, NULL);

	return tv.tv_sec * 1000 + tv.tv_usec/1000.0;
}

static double 
SubtractTime( long uiEndTime, long uiStartTime )
{
	return uiEndTime - uiStartTime;
}

////////////////////////////////////////////////////////////////////////////////

static int LoadTextFromFile(
	const char *file_name, char **result_string, size_t *string_len)
{
	int fd;
	unsigned file_len;
	struct stat file_status;
	int ret;

	*string_len = 0;
	fd = open(file_name, O_RDONLY);
	if (fd == -1)
	{
		printf("Error opening file %s\n", file_name);
		return -1;
	}
	ret = fstat(fd, &file_status);
	if (ret)
	{
		printf("Error reading status for file %s\n", file_name);
		return -1;
	}
	file_len = file_status.st_size;

	*result_string = (char*)calloc(file_len + 1, sizeof(char));
	ret = read(fd, *result_string, file_len);
	if (!ret)
	{
		printf("Error reading from file %s\n", file_name);
		return -1;
	}

	close(fd);

	*string_len = file_len;
	return 0;
}

static void DrawString(float x, float y, float color[4], char *buffer)
{
	unsigned int uiLen, i;

	glPushAttrib(GL_LIGHTING_BIT);
	glDisable(GL_LIGHTING);

	glRasterPos2f(x, y);
	glColor3f(color[0], color[1], color[2]);
	uiLen = (unsigned int) strlen(buffer);
	for (i = 0; i < uiLen; i++)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, buffer[i]);
	}
	glPopAttrib();
}

static void DrawText(float x, float y, int light, const char *format, ...)
{
	va_list args;
	char buffer[256];
	GLint iVP[4];
	GLint iMatrixMode;

	va_start(args, format);
	vsprintf(buffer, format, args);
	va_end(args);

	glPushAttrib(GL_LIGHTING_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);

	glGetIntegerv(GL_VIEWPORT, iVP);
	glViewport(0, 0, WindowWidth, WindowHeight);
	glGetIntegerv(GL_MATRIX_MODE, &iMatrixMode);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glScalef(2.0f / WindowWidth, -2.0f / WindowHeight, 1.0f);
	glTranslatef(-WindowWidth / 2.0f, -WindowHeight / 2.0f, 0.0f);

	if(light)
	{
		glColor4fv(ShadowTextColor);
		DrawString(x-0, y-0, ShadowTextColor, buffer);

		glColor4fv(HighlightTextColor);
		DrawString(x-2, y-2, HighlightTextColor, buffer);
	}
	else
	{
		glColor4fv(HighlightTextColor);
		DrawString(x-0, y-0, HighlightTextColor, buffer);

		glColor4fv(ShadowTextColor);
		DrawString(x-2, y-2, ShadowTextColor, buffer);   
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);

	glPopMatrix();
	glMatrixMode(iMatrixMode);

	glPopAttrib();
	glViewport(iVP[0], iVP[1], iVP[2], iVP[3]);
}


static float
RandomFloat(float randMax, float randMin)
{
	float result;
	result =(float)rand() / (float)RAND_MAX;

	return ((1.0f - result) * randMin + result *randMax);
}

static int 
InitData()
{
	// make sure DataParticleCount is multiple of group size
	DataParticleCount = DataParticleCount < GroupSize ? GroupSize :
	DataParticleCount;
	DataParticleCount = (DataParticleCount / GroupSize) * GroupSize;

	DataBodyCount = DataParticleCount;

	if (DataInput)
		free(DataInput);
	DataInput = (float *)calloc(1, DataBodyCount * sizeof(cl_float4));

	// initialization of inputs
	for(int i = 0; i < DataBodyCount; ++i)
	{
		int index = 4 * i;

		// First 3 values are position in x,y and z direction
		for(int j = 0; j < 3; ++j)
		{
			DataInput[index + j] = RandomFloat(3, 50);
		}

		// Mass value
		DataInput[index + 3] = RandomFloat(1, 1000);
	}

	return 1;
}


#if USE_GL_ATTACHMENTS
#else
static int
UpdateVBO(int index, void *data, int gl_attrib_array_index)
{
	if (VboPosID[index] && data)
	{
		glBindBuffer(GL_ARRAY_BUFFER, VboPosID[index]);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * DataBodyCount, data, GL_STATIC_DRAW);
		glVertexAttribPointer(gl_attrib_array_index, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(gl_attrib_array_index);
	}
	else
	{
		printf("Invalid VboPosID[%d]\n", index);
		return -1;
	}

	return 1;
}
#endif

static int
CreateGLResouce()
{
	GLint bsize;

	// VAO
	glGenVertexArrays(1, &VaoID);
	glBindVertexArray(VaoID);
	if (!VaoID)
	{
		printf("VAO generating failed!\n");
		return -1;
	}

	// VBOPOS0
	if (VboPosID[0])
		glDeleteBuffers(1, &VboPosID[0]);
	glGenBuffers(1, &VboPosID[0]);
	if (!VboPosID[0])
	{
		printf("VBO VboPosID[0] generation failed\n");
		return -1;
	}
	glBindBuffer(GL_ARRAY_BUFFER, VboPosID[0]);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * DataBodyCount, DataInput, GL_STATIC_DRAW);
	glVertexAttribPointer(VboPosLoc[0], 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(VboPosLoc[0]);

	// recheck the size of the created buffer to make sure its what we requested
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize); 
	if ((GLuint)bsize != 4 * sizeof(float) * DataBodyCount) {
		printf("Vertex Buffer object (%d) has incorrect size (%d).\n", VboPosID[0], bsize);
	}

	// VBOPOS1
	if (VboPosID[1])
		glDeleteBuffers(1, &VboPosID[1]);
	glGenBuffers(1, &VboPosID[1]);
	if (!VboPosID[1])
	{
		printf("VBO VboPosID[1] generation failed\n");
		return -1;
	}
	glBindBuffer(GL_ARRAY_BUFFER, VboPosID[1]);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * DataBodyCount, DataInput, GL_STATIC_DRAW);
	glVertexAttribPointer(VboPosLoc[1], 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(VboPosLoc[1]);

	// recheck the size of the created buffer to make sure its what we requested
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize); 
	if ((GLuint)bsize != 4 * sizeof(float) * DataBodyCount) {
		printf("Vertex Buffer object (%d) has incorrect size (%d).\n", VboPosID[1], bsize);
	}

	GLenum error_check_value = glGetError();
	if (error_check_value != GL_NO_ERROR)
	{
		fprintf(stderr, "error: could not create VBO: %s\n",
			gluErrorString(error_check_value));
		exit(1);
	}

	return 1;
}

static int
Recompute(void)
{
	if(!ComputeKernel)
		return CL_SUCCESS;

	int err = 0;

	int currentBuffer = CurrentBuffer;
	int nextBuffer = (CurrentBuffer+1)%2;

	if(Animated || Update)
	{

		glFinish();

        // If use shared context, then data should be already in GL VBOs even for the 1st frame
#if (USE_GL_ATTACHMENTS)

		err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &ComputePosBuffer[currentBuffer], 0, 0, 0);
		if (err != CL_SUCCESS)
		{
			printf("Failed to acquire GL object! %d\n", err);
			return EXIT_FAILURE;
		}

		err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &ComputePosBuffer[nextBuffer], 0, 0, 0);
		if (err != CL_SUCCESS)
		{
			printf("Failed to acquire GL object! %d\n", err);
			return EXIT_FAILURE;
		}

#if (DEBUG_INFO)
		float *DataCurPos = (float *)calloc(1, 4 * sizeof(float) * DataBodyCount);
		float *DataNexPos = (float *)calloc(1, 4 * sizeof(float) * DataBodyCount);

		err = clEnqueueReadBuffer( ComputeCommands, ComputePosBuffer[currentBuffer], CL_TRUE, 0, 4 * sizeof(float) * DataBodyCount, DataCurPos, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		err = clEnqueueReadBuffer( ComputeCommands, ComputePosBuffer[nextBuffer], CL_TRUE, 0, 4 * sizeof(float) * DataBodyCount, DataNexPos, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		for (int i = 0; i < DataBodyCount; ++i)
			printf("Before NDRange %d - %d: org [%f %f %f %f] - org curr [%f %f %f %f] - org next [%f %f %f %f]\n", NDRangeCount, i, 
				DataInput[4 * i], DataInput[4 * i + 1], DataInput[4 * i + 2], DataInput[4 * i + 3],
				DataCurPos[4 * i], DataCurPos[4 * i + 1], DataCurPos[4 * i + 2], DataCurPos[4 * i + 3],
				DataNexPos[4 * i], DataNexPos[4 * i + 1], DataNexPos[4 * i + 2], DataNexPos[4 * i + 3]);

		free(DataCurPos);
		free(DataNexPos);
#endif

#else
        // Not sharing context with OpenGL, needs to explicitly send data to GPU for the 1st frame
		if (!NDRangeCount)
		{
			printf("1st Frame! Let's send data to GPU!\n");
			err = clEnqueueWriteBuffer(ComputeCommands, ComputePosBuffer[currentBuffer], 1, 0, 
				4 * sizeof(float) * DataBodyCount, DataInput, 0, 0, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Failed to write buffer! %d\n", err);
				return EXIT_FAILURE;
			}

			err = clEnqueueWriteBuffer(ComputeCommands, ComputePosBuffer[nextBuffer], 1, 0, 
				4 * sizeof(float) * DataBodyCount, DataInput, 0, 0, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Failed to write buffer! %d\n", err);
				return EXIT_FAILURE;
			}
		}

#endif
		Update = 0;
		err = CL_SUCCESS;
		err |= clSetKernelArg(ComputeKernel, 0, sizeof(cl_mem), &ComputePosBuffer[currentBuffer]);
		err |= clSetKernelArg(ComputeKernel, 1, sizeof(cl_mem), &ComputeVelBuffer[currentBuffer]);
		err |= clSetKernelArg(ComputeKernel, 5, sizeof(cl_mem), &ComputePosBuffer[nextBuffer]);
		err |= clSetKernelArg(ComputeKernel, 6, sizeof(cl_mem), &ComputeVelBuffer[nextBuffer]);
		if (err)
			return -10;

		size_t global[1];
		size_t local[1];

		global[0] = DataBodyCount;
		local[0] = GroupSize;

#if (DEBUG_INFO)
		if(FrameCount <= 1)
			printf("Global[%4d] Local[%4d]\n", 
				(int)global[0], (int)local[0]);
#endif

		err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel, 1, NULL, global, local, 0, NULL, NULL);
		if (err)
		{
			printf("Failed to enqueue kernel! %d\n", err);
			return err;
		}

		NDRangeCount++;

#if (DEBUG_INFO)

		float *DataCurPos = (float *)calloc(1, 4 * sizeof(float) * DataBodyCount);
		float *DataNexPos = (float *)calloc(1, 4 * sizeof(float) * DataBodyCount);

		err = clEnqueueReadBuffer( ComputeCommands, ComputePosBuffer[currentBuffer], CL_TRUE, 0, 4 * sizeof(float) * DataBodyCount, DataCurPos, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		err = clEnqueueReadBuffer( ComputeCommands, ComputePosBuffer[nextBuffer], CL_TRUE, 0, 4 * sizeof(float) * DataBodyCount, DataNexPos, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		for (int i = 0; i < DataBodyCount; ++i)
			printf("After NDRange %d - %d: org [%f %f %f %f] - org curr [%f %f %f %f] - org next [%f %f %f %f]\n", NDRangeCount, i, 
				DataInput[4 * i], DataInput[4 * i + 1], DataInput[4 * i + 2], DataInput[4 * i + 3],
				DataCurPos[4 * i], DataCurPos[4 * i + 1], DataCurPos[4 * i + 2], DataCurPos[4 * i + 3],
				DataNexPos[4 * i], DataNexPos[4 * i + 1], DataNexPos[4 * i + 2], DataNexPos[4 * i + 3]);
		
		free(DataCurPos);
		free(DataNexPos);
#endif

#if (USE_GL_ATTACHMENTS)

        // Release control and the data is already in VBOs
		err = clEnqueueReleaseGLObjects(ComputeCommands, 1, &ComputePosBuffer[currentBuffer], 0, 0, 0);
		if (err != CL_SUCCESS)
		{
			printf("Failed to release GL object! %d\n", err);
			return EXIT_FAILURE;
		}

		err = clEnqueueReleaseGLObjects(ComputeCommands, 1, &ComputePosBuffer[nextBuffer], 0, 0, 0);
		if (err != CL_SUCCESS)
		{
			printf("Failed to release GL object! %d\n", err);
			return EXIT_FAILURE;
		}

#else
        // Explicitly copy data back to host
		err = clEnqueueReadBuffer( ComputeCommands, ComputePosBuffer[nextBuffer], CL_TRUE, 0, DataParticleCount * sizeof(float), DataInput, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

        // Data in host side, copy to VBOs
		UpdateVBO(nextBuffer, DataInput, nextBuffer);
#endif

		clFinish(ComputeCommands);
	}

	// Notify GL side which attribute index is using
	glUniform1i(UniformCurBufIdxLocation, nextBuffer);

	// Switch buffers
	CurrentBuffer = nextBuffer;

	return CL_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////

static int 
CreateComputeResource(void)
{
	int err = 0;

#if (USE_GL_ATTACHMENTS)

	// CL Context is created from GL context, GL VBOs and CL Buffers point to the same data in GPU memory
	// Updating VBO or associated CL Buffer affects both CL and GL.

	// CL buffer Pos 0
	if(ComputePosBuffer[0])
		clReleaseMemObject(ComputePosBuffer[0]);
	ComputePosBuffer[0] = 0;

	if (VboPosID[0])
	{
		printf("Allocating compute input/output real part for FFT in device memory...\n");
		ComputePosBuffer[0] = clCreateFromGLBuffer(ComputeContext, CL_MEM_READ_WRITE, VboPosID[0], &err);
		if (!ComputePosBuffer[0] || err != CL_SUCCESS)
		{
			printf("Failed to create OpenGL VBO reference! %d\n", err);
			return -1;
		}
	}
	else
	{
		printf("VboPosID[0] not valid!\n");
		return -1;
	}

	// CL buffer Pos 1
	if(ComputePosBuffer[1])
		clReleaseMemObject(ComputePosBuffer[1]);
	ComputePosBuffer[1] = 0;

	if (VboPosID[1])
	{
		printf("Allocating compute input/output real part for FFT in device memory...\n");
		ComputePosBuffer[1] = clCreateFromGLBuffer(ComputeContext, CL_MEM_READ_WRITE, VboPosID[1], &err);
		if (!ComputePosBuffer[1] || err != CL_SUCCESS)
		{
			printf("Failed to create OpenGL VBO reference! %d\n", err);
			return -1;
		}
	}
	else
	{
		printf("VboPosID[1] not valid!\n");
		return -1;
	}

#else

    // Not sharing context, so just create CL buffers as normal
	if(ComputePosBuffer[0])
		clReleaseMemObject(ComputePosBuffer[0]);
	ComputePosBuffer[0] = 0;

	printf("Allocating compute buffer 0 for NBody in device memory...\n");
	ComputePosBuffer[0] = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE,
		4 * sizeof(float) * DataBodyCount, 0, &err);
	if (!ComputePosBuffer[0] || err != CL_SUCCESS)
	{
		printf("Failed to create OpenGL VBO reference! %d\n", err);
		return -1;
	}

	if(ComputePosBuffer[1])
		clReleaseMemObject(ComputePosBuffer[1]);
	ComputePosBuffer[1] = 0;

	printf("Allocating compute buffer 1 for NBody in device memory...\n");
	ComputePosBuffer[1] = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE,
		4 * sizeof(float) * DataBodyCount, 0, &err);
	if (!ComputePosBuffer[1] || err != CL_SUCCESS)
	{
		printf("Failed to create OpenGL VBO reference! %d\n", err);
		return -1;
	}

#endif

	// Velocity buffer 0
	if(ComputeVelBuffer[0])
		clReleaseMemObject(ComputeVelBuffer[0]);
	ComputeVelBuffer[0] = 0;

	printf("Allocating compute velocity buffer 0 for NBody in device memory...\n");
	ComputeVelBuffer[0] = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE,
		4 * sizeof(float) * DataBodyCount, 0, &err);
	if (!ComputeVelBuffer[0] || err != CL_SUCCESS)
	{
		printf("Failed to create OpenGL VBO reference! %d\n", err);
		return -1;
	}

	// Velocity buffer 1
	if(ComputeVelBuffer[1])
		clReleaseMemObject(ComputeVelBuffer[1]);
	ComputeVelBuffer[1] = 0;

	printf("Allocating compute velocity buffer 0 for NBody in device memory...\n");
	ComputeVelBuffer[1] = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE,
		4 * sizeof(float) * DataBodyCount, 0, &err);
	if (!ComputeVelBuffer[1] || err != CL_SUCCESS)
	{
		printf("Failed to create OpenGL VBO reference! %d\n", err);
		return -1;
	}

	// Initialize the velocity buffer to zero
	float* p = (float*) clEnqueueMapBuffer(ComputeCommands, ComputeVelBuffer[0], CL_TRUE,
		CL_MAP_WRITE
		, 0, 4 * sizeof(float) * DataBodyCount, 0, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error mapping ComputeVelBuffer[0]\n");
		exit(-1);
	}
	memset(p, 0, 4 * sizeof(float) * DataBodyCount);
	err = clEnqueueUnmapMemObject(ComputeCommands, ComputeVelBuffer[0], p, 0, NULL,NULL);

	p = (float*) clEnqueueMapBuffer(ComputeCommands, ComputeVelBuffer[1], CL_TRUE,
		CL_MAP_WRITE
		, 0, 4 * sizeof(float) * DataBodyCount, 0, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error mapping ComputeVelBuffer[1]\n");
		exit(-1);
	}
	memset(p, 0, 4 * sizeof(float) * DataBodyCount);
	err = clEnqueueUnmapMemObject(ComputeCommands, ComputeVelBuffer[1], p, 0, NULL,NULL);

	return CL_SUCCESS;
}

static int 
SetupComputeDevices(int gpu)
{
	int err;
	size_t returned_size;
	ComputeDeviceType = gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

#if (USE_GL_ATTACHMENTS)

	printf(SEPARATOR);
	printf("Using active OpenGL context...\n");

    // Bind to platform
	cl_platform_id platform_id;

	cl_uint numPlatforms;
	cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		printf("clGetPlatformIDs Failed\n");
		return EXIT_FAILURE;
	}

	if (0 < numPlatforms)
	{
		cl_platform_id* platforms = (cl_platform_id*)calloc(numPlatforms, sizeof(cl_platform_id));

		status = clGetPlatformIDs(numPlatforms, platforms, NULL);

		char platformName[100];
		for (unsigned i = 0; i < numPlatforms; ++i)
		{
			status = clGetPlatformInfo(platforms[i],
				CL_PLATFORM_VENDOR,
				sizeof(platformName),
				platformName,
				NULL);
			platform_id = platforms[i];
			if (!strcmp(platformName, "Advanced Micro Devices, Inc."))
			{
				break;
			}
		}
		printf("Platform found : %s\n", platformName);
		free(platforms);
	}
	if(NULL == platform_id)
	{
		printf("NULL platform found so Exiting Application.\n");
		return EXIT_FAILURE;
	}

    // Get ID for the device
	err = clGetDeviceIDs(platform_id, ComputeDeviceType, 1, &ComputeDeviceId, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to locate compute device!\n");
		return EXIT_FAILURE;
	}

    // Create a context  
	cl_context_properties properties[] =
	{
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
		CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_id),
		0
	};

    // Create a context from a CGL share group
    //
	ComputeContext = clCreateContext(properties, 1, &ComputeDeviceId, NULL, 0, 0);
	if (!ComputeContext)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

#else

    // Bind to platform
	cl_platform_id platform_id;

	cl_uint numPlatforms;
	cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		printf("clGetPlatformIDs Failed\n");
		return EXIT_FAILURE;
	}

	if (0 < numPlatforms)
	{
		cl_platform_id* platforms = (cl_platform_id*)calloc(numPlatforms, sizeof(cl_platform_id));

		status = clGetPlatformIDs(numPlatforms, platforms, NULL);

		char platformName[100];
		for (unsigned i = 0; i < numPlatforms; ++i)
		{
			status = clGetPlatformInfo(platforms[i],
				CL_PLATFORM_VENDOR,
				sizeof(platformName),
				platformName,
				NULL);
			platform_id = platforms[i];
			if (!strcmp(platformName, "Advanced Micro Devices, Inc."))
			{
				break;
			}
		}
		printf("Platform found : %s\n", platformName);
		free(platforms);
	}
	if(NULL == platform_id)
	{
		printf("NULL platform found so Exiting Application.\n");
		return EXIT_FAILURE;
	}

    // Get ID for the device
	err = clGetDeviceIDs(platform_id, ComputeDeviceType, 1, &ComputeDeviceId, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to locate compute device!\n");
		return EXIT_FAILURE;
	}

    // Create a context containing the compute device(s)
    //
	ComputeContext = clCreateContext(0, 1, &ComputeDeviceId, NULL, NULL, &err);
	if (!ComputeContext)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

#endif

	unsigned int device_count;
	cl_device_id device_ids[16];

	err = clGetContextInfo(ComputeContext, CL_CONTEXT_DEVICES, sizeof(device_ids), device_ids, &returned_size);
	if(err)
	{
		printf("Error: Failed to retrieve compute devices for context!\n");
		return EXIT_FAILURE;
	}

	device_count = returned_size / sizeof(cl_device_id);

	unsigned int i = 0;
	int device_found = 0;
	cl_device_type device_type; 
	for(i = 0; i < device_count; i++) 
	{
		clGetDeviceInfo(device_ids[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
		if(device_type == ComputeDeviceType) 
		{
			ComputeDeviceId = device_ids[i];
			device_found = 1;
			break;
		} 
	}

	if(!device_found)
	{
		printf("Error: Failed to locate compute device!\n");
		return EXIT_FAILURE;
	}

    // Create a command queue
    //
	ComputeCommands = clCreateCommandQueue(ComputeContext, ComputeDeviceId, 0, &err);
	if (!ComputeCommands)
	{
		printf("Error: Failed to create a command queue!\n");
		return EXIT_FAILURE;
	}

    // Report the device vendor and device name
    // 
	cl_char vendor_name[1024] = {0};
	cl_char device_name[1024] = {0};
	err = clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
	err|= clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to retrieve device info!\n");
		return EXIT_FAILURE;
	}

	printf(SEPARATOR);
	printf("Connecting to %s %s...\n", vendor_name, device_name);

	return CL_SUCCESS;
}

static int
SetupGLProgram()
{
	GLenum error_check_value = glGetError();

	if(VertexShaderID)
		glDeleteShader(VertexShaderID);
	VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(VertexShaderID, 1, &VertexShaderSource, NULL);
	glCompileShader(VertexShaderID);

	if(FragShaderID)
		glDeleteShader(FragShaderID);
	FragShaderID = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(FragShaderID, 1, &FragShaderSource, NULL);
	glCompileShader(FragShaderID);

	if (GLProgramID)
	{
		glUseProgram(0);
		glDeleteProgram(GLProgramID);
	}
	GLProgramID = glCreateProgram();
	glAttachShader(GLProgramID, VertexShaderID);
	glAttachShader(GLProgramID, FragShaderID);

	glLinkProgram(GLProgramID);
	glUseProgram(GLProgramID);

	VboPosLoc[0] = glGetAttribLocation(GLProgramID, "pos0");
	VboPosLoc[1] = glGetAttribLocation(GLProgramID, "pos1");

	printf("%d %d\n", VboPosLoc[0], VboPosLoc[1]);

	UniformCurBufIdxLocation = glGetUniformLocation(GLProgramID, "curBufIdx");
	if(UniformCurBufIdxLocation == 0xFFFFFFFF)
		return -1;

	error_check_value = glGetError();
	if (error_check_value != GL_NO_ERROR)
	{
		fprintf(stderr, "error: %s: %s\n",
			__FUNCTION__,
			gluErrorString(error_check_value));
		exit(1);
	}

	return 1;
}

static int 
SetupComputeKernel(void)
{
	int err = 0;
	char *source = 0;
	size_t length = 0;

	if(ComputeKernel)
		clReleaseKernel(ComputeKernel);    
	ComputeKernel = 0;

	if(ComputeProgram)
		clReleaseProgram(ComputeProgram);
	ComputeProgram = 0;

	printf(SEPARATOR);
	printf("Loading kernel source from file '%s'...\n", COMPUTE_KERNEL_FILENAME);    
	err = LoadTextFromFile(COMPUTE_KERNEL_FILENAME, &source, &length);
	if (!source || err)
	{
		printf("Error: Failed to load kernel source!\n");
		return EXIT_FAILURE;
	}

#if (DEBUG_INFO)
	printf("%s", source);
#endif

    // Create the compute program from the source buffer
    //
	ComputeProgram = clCreateProgramWithSource(ComputeContext, 1, (const char **) & source, NULL, &err);
	if (!ComputeProgram || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}
	free(source);

    // Build the program executable
    //
    // err = clBuildProgram(ComputeProgram, 0, NULL, "-x clc++", NULL, NULL);
	err = clBuildProgram(ComputeProgram, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(ComputeProgram, ComputeDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		return EXIT_FAILURE;
	}

    // Create the compute kernel from within the program
    //
	printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_MATMUL_NAME); 
	ComputeKernel = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_MATMUL_NAME, &err);

	if (!ComputeKernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		return EXIT_FAILURE;
	}

    // Get the maximum work group size for executing the kernel on the device
    //
	err = clGetKernelWorkGroupInfo(ComputeKernel, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &MaxWorkGroupSize, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to retrieve kernel work group info! %d\n", err);
		exit(1);
	}

#if (DEBUG_INFO)
	printf("MaxWorkGroupSize: %d\n", MaxWorkGroupSize);
	printf("WorkGroupItems: %d\n", WorkGroupItems);
#endif

	WorkGroupSize[0] = (MaxWorkGroupSize > 1) ? (MaxWorkGroupSize / WorkGroupItems) : MaxWorkGroupSize;
    // WorkGroupSize[1] = MaxWorkGroupSize / WorkGroupSize[0];

	printf(SEPARATOR);

	// Setup several arguments that won't change
    // DataBodyCount
	err = clSetKernelArg(
		ComputeKernel,
		2,
		sizeof(int),
		(void *)&DataBodyCount);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arg 2: DataBodyCount! %d\n", err);
		exit(1);
	}

    // time step
	err = clSetKernelArg(
		ComputeKernel,
		3,
		sizeof(float),
		(void *)&delT);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arg 3: delT! %d\n", err);
		exit(1);
	}

    // upward Pseudoprobability
	err = clSetKernelArg(
		ComputeKernel,
		4,
		sizeof(float),
		(void *)&espSqr);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arg 4: espSqr! %d\n", err);
		exit(1);
	}

	return CL_SUCCESS;
}

static void
Cleanup(void)
{
	clFinish(ComputeCommands);
	clReleaseKernel(ComputeKernel);
	clReleaseProgram(ComputeProgram);
	clReleaseCommandQueue(ComputeCommands);
	clReleaseMemObject(ComputePosBuffer[0]);
	clReleaseMemObject(ComputePosBuffer[1]);
	clReleaseMemObject(ComputeVelBuffer[0]);
	clReleaseMemObject(ComputeVelBuffer[1]);
	clReleaseContext(ComputeContext);

	ComputeCommands = 0;
	ComputeKernel = 0;
	ComputeProgram = 0;    
	ComputePosBuffer[0] = 0;
	ComputePosBuffer[1] = 0;
	ComputeContext = 0;

	if (DataInput)
		free(DataInput);
}

static void
Shutdown(void)
{   fclose(fp);
	printf(SEPARATOR);
	printf("Shutting down...\n");
	Cleanup();
	exit(0);
}

////////////////////////////////////////////////////////////////////////////////

static int 
SetupGraphics(void)
{
	GLenum GlewInitResult;

	GlewInitResult = glewInit();
	if (GlewInitResult != GLEW_OK)
	{
		fprintf(stderr, "ERROR: %s\n", glewGetErrorString(GlewInitResult));
		exit(1);
	}

	glClearColor (0.0, 0.0, 0.0, 0.0);

	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, WindowWidth, WindowHeight);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	return GL_NO_ERROR;
}

static int 
Initialize(int gpu)
{
	int err;
	err = SetupGraphics();
	if (err != GL_NO_ERROR)
	{
		printf ("Failed to setup OpenGL state!");
		exit (err);
	}

	err = SetupComputeDevices(gpu);
	if(err != CL_SUCCESS)
	{
		printf ("Failed to connect to compute device! Error %d\n", err);
		exit (err);
	}

	err = InitData();
	if (err != 1)
	{
		printf ("Failed to Init FFT Data! Error %d\n", err);
		exit (err);
	}

	err = SetupGLProgram();
	if (err != 1)
	{
		printf ("Failed to setup OpenGL Shader! Error %d\n", err);
		exit (err);
	}

	err = CreateGLResouce();
	if (err != 1)
	{
		printf ("Failed to create GL resource! Error %d\n", err);
		exit (err);
	}

	err = SetupComputeKernel();
	if (err != CL_SUCCESS)
	{
		printf ("Failed to setup compute kernel! Error %d\n", err);
		exit (err);
	}

	err = CreateComputeResource();
	if(err != CL_SUCCESS)
	{
		printf ("Failed to create compute result! Error %d\n", err);
		exit (err);
	}

	return CL_SUCCESS;
}


static void
ReportInfo(void)
{
	if(ShowStats)
	{
		int iX = 20;
		int iY = 20;

		DrawText(iX - 1, WindowHeight - iY - 1, 0, StatsString);
		DrawText(iX - 2, WindowHeight - iY - 2, 0, StatsString);
		DrawText(iX, WindowHeight - iY, 1, StatsString);
	}

	if(ShowInfo)
	{
		int iX = TextOffset[0];
		int iY = WindowHeight - TextOffset[1];

		DrawText(WindowWidth - iX - 1 - strlen(InfoString) * 10, WindowHeight - iY - 1, 0, InfoString);
		DrawText(WindowWidth - iX - 2 - strlen(InfoString) * 10, WindowHeight - iY - 2, 0, InfoString);
		DrawText(WindowWidth - iX - strlen(InfoString) * 10, WindowHeight - iY, 1, InfoString);

		ShowInfo = (ShowInfo > 200) ? 0 : ShowInfo + 1;
	}
}

static void 
ReportStats(
	uint64_t uiStartTime, uint64_t uiEndTime)
{
	TimeElapsed += SubtractTime(uiEndTime, uiStartTime);

	if(TimeElapsed && FrameCount && FrameCount > (int)ReportStatsInterval) 
	{
		double fMs = (TimeElapsed / (double) FrameCount);
		double fFps = 1.0 / (fMs / 1000.0);

		sprintf(StatsString, "[%s] Compute: %3.2f ms  Display: %3.2f fps (%s)\n", 
			(ComputeDeviceType == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU", 
			fMs, fFps, USE_GL_ATTACHMENTS ? "attached" : "copying");

		glutSetWindowTitle(StatsString);
		fprintf(fp,"%s", StatsString);
		FrameCount = 0;
		TimeElapsed = 0;
	}    
}

static void
Display_(void)
{
	FrameCount++;
	uint64_t uiStartTime = GetCurrentTime();

	glClearColor (0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	if(Animated || Update)
	{
		int err = Recompute();
		if (err != 0)
		{
			printf("Error %d from Recompute!\n", err);
			exit(1);
		}

	}

	// glBindVertexArray(VaoID);
	glDrawArrays(GL_POINTS, 0, DataBodyCount);
	ReportInfo();

	glFinish(); // for timing

	uint64_t uiEndTime = GetCurrentTime();
	ReportStats(uiStartTime, uiEndTime);
	DrawText(TextOffset[0], TextOffset[1], 1, (Animated == 0) ? "Press space to animate" : " ");
	glutSwapBuffers();
}

static void 
Reshape (int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glClear(GL_COLOR_BUFFER_BIT);
	glutSwapBuffers();

	if(w > 2 * WindowWidth || h > 2 * WindowHeight)
	{
		WindowWidth = w;
		WindowHeight = h;
		Cleanup();
		if(Initialize(ComputeDeviceType == CL_DEVICE_TYPE_GPU) != GL_NO_ERROR)
			Shutdown();
	}

	WindowWidth = w;
	WindowHeight = h;    
}

void Keyboard( unsigned char key, int x, int y )
{
	switch( key )
	{
		case 27:
		exit(0);
		break;

		case ' ':
		Animated = !Animated;
		sprintf(InfoString, "Animated = %s\n", Animated ? "true" : "false");
		ShowInfo = 1;
		break;

		case 'i':
		ShowInfo = ShowInfo > 0 ? 0 : 1;
		break;

		case 's':
		ShowStats = ShowStats > 0 ? 0 : 1;
		break;
	}
	Update = 1;
	glutPostRedisplay();
}

void Idle(void)
{
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
    // Parse command line options
    //
	int i;
	int use_gpu = 1;
	for( i = 0; i < argc && argv; i++)
	{
		if(!argv[i])
			continue;

		if(strstr(argv[i], "cpu"))
			use_gpu = 0;        

		else if(strstr(argv[i], "gpu"))
			use_gpu = 1;
	}

	fp = fopen("fft_res", "w+");
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (WindowWidth, WindowHeight);
	glutInitWindowPosition (100, 100);
	glutCreateWindow (argv[0]);
	if (Initialize (use_gpu) == GL_NO_ERROR)
	{
		glutDisplayFunc(Display_);
		glutIdleFunc(Idle);
		glutReshapeFunc(Reshape);
		glutKeyboardFunc(Keyboard);

		atexit(Shutdown);
		printf("Starting event loop...\n");

		glutMainLoop();
	}

	return 0;
}

