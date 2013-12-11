////////////////////////////////////////////////////////////////////////////////////////////////////
//
// OpenCL Kernel taken from AMDAPPSDK
// Program Structure from APPLE QJulia
// Remixed by Xiang
//
////////////////////////////////////////////////////////////////////////////////////////////////////

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

#define USE_GL_ATTACHMENTS              (1)  // enable OpenGL attachments for Compute results
#define DEBUG_INFO                      (0)     
#define COMPUTE_KERNEL_FILENAME         ("FFT_Kernels.cl")
#define COMPUTE_KERNEL_MATMUL_NAME      ("kfft")
#define SEPARATOR                       ("----------------------------------------------------------------------\n")

////////////////////////////////////////////////////////////////////////////////

#define DATA_REAL_MIN                   (0.0)
#define DATA_REAL_MAX                   (10.0)
#define DATA_IMAG_MIN                   (0.0)
#define DATA_IMAG_MAX                   (10.0)

////////////////////////////////////////////////////////////////////////////////

static GLuint                            VboRealID;
static GLuint                            VboImaginnaryID;
static GLuint                            VaoID;
static GLuint                            VertexShaderID;
static GLuint                            FragShaderID;
static GLuint                            GLProgramID;

////////////////////////////////////////////////////////////////////////////////

const char *VertexShaderSource = 
	"#version 430\n"

	"layout(location=0) in vec4 fft_real;\n"
	"layout(location=1) in vec4 fft_imaginary;\n"
	"out vec4 ex_Color;\n"

	"void main(void)\n"
	"{\n"
	"	gl_Position = fft_imaginary / fft_real;\n"
	"	ex_Color = vec4(1.0, 1.0, 1.0, 1.0);\n"
	"}\n";

const char *FragShaderSource = 
	"#version 430\n"

	"in vec4 ex_Color;\n"
	"out vec4 out_Color;\n"

	"void main(void)\n"
	"{\n"
	"	out_Color = ex_Color;\n"
	"}\n";

////////////////////////////////////////////////////////////////////////////////

static cl_context                       ComputeContext;
static cl_command_queue                 ComputeCommands;
static cl_kernel                        ComputeKernel;
static cl_program                       ComputeProgram;
static cl_device_id                     ComputeDeviceId;
static cl_device_type                   ComputeDeviceType;
static cl_mem                           ComputeInputOutputReal;
static cl_mem                           ComputeInputOutputImaginary;

////////////////////////////////////////////////////////////////////////////////

static int Animated                     = 0;
static int Update                       = 1;

static int Width                        = 128;
static int Height                       = 128;

static float *DataReal                  = NULL;
static float *DataImaginary             = NULL;

static int DataWidth                    = Width;
static int DataHeight                   = Height;
static int DataElemCount                = DataWidth * DataHeight;

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

static float VertexPos[4][2]            = { { -1.0f, -1.0f },
{ +1.0f, -1.0f },
{ +1.0f, +1.0f },
{ -1.0f, +1.0f } };

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
	glViewport(0, 0, Width, Height);
	glGetIntegerv(GL_MATRIX_MODE, &iMatrixMode);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glScalef(2.0f / Width, -2.0f / Height, 1.0f);
	glTranslatef(-Width / 2.0f, -Height / 2.0f, 0.0f);

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

static void
RandomFillArray_Float(float *arrayPtr, int width, int height, float rangeMin, float rangeMax)
{
	if (arrayPtr)
	{
		unsigned int seed = (unsigned int)GetCurrentTime();

		srand(seed);
		double range = double(rangeMax - rangeMin) + 1.0;

		for(int i = 0; i < height; i++)
			for(int j = 0; j < width; j++)
			{
				int index = i*width + j;
				arrayPtr[index] = rangeMin + float(range*rand()/(RAND_MAX + 1.0));
			}
		}
	}

static float *
CreateRandomFilledArray_Float(int width, int height, float rangeMin, float rangeMax)
{
	float *array;

	array = (float *)calloc(1, width * height * sizeof(float));

	RandomFillArray_Float(array, width, height, rangeMin, rangeMax);

	return array;
}

static int 
InitData()
{
	if (DataReal)
		free(DataReal);
	DataReal = CreateRandomFilledArray_Float(Width, Height, DATA_REAL_MIN, DATA_REAL_MAX);

	if (DataImaginary)
		free(DataImaginary);
	DataImaginary = CreateRandomFilledArray_Float(Width, Height, DATA_IMAG_MIN, DATA_IMAG_MAX);

	return 1;
}

static int
UpdateData()
{
	RandomFillArray_Float(DataReal, Width, Height, 0.0, 100.0);
	RandomFillArray_Float(DataImaginary, Width, Height, 0.0, 100.0);

	return 1;
}

static int
UpdateVBOs()
{
		if (VboRealID)
		{
			glBindBuffer(GL_ARRAY_BUFFER, VboRealID);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * DataElemCount, DataReal, GL_STATIC_DRAW);
			glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(0);
		}
		else
		{
			printf("Invalid VboRealID\n");
			return -1;
		}

		if (VboImaginnaryID)
		{
			glBindBuffer(GL_ARRAY_BUFFER, VboImaginnaryID);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * DataElemCount, DataImaginary, GL_STATIC_DRAW);
			glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(0);
		}
		else
		{
			printf("Invalid VboRealID\n");
			return -1;
		}

		return 1;
}

static int
CreateGLResouce()
{
	glGenVertexArrays(1, &VaoID);
	glBindVertexArray(VaoID);
	if (!VaoID)
	{
		printf("VAO generating failed!\n");
		return -1;
	}

	if (VboRealID)
		glDeleteBuffers(1, &VboRealID);
	glGenBuffers(1, &VboRealID);
	if (!VboRealID)
	{
		printf("VBO VboRealID generating failed\n");
		return -1;
	}
	glBindBuffer(GL_ARRAY_BUFFER, VboRealID);
	if (DataReal)
	{
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * DataElemCount, DataReal, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);
	}
	else
	{
		printf("DataReal is not ready!\n");
		return -1;
	}

	if (VboImaginnaryID)
		glDeleteBuffers(1, &VboImaginnaryID);
	glGenBuffers(1, &VboImaginnaryID);
	if (!VboImaginnaryID)
	{
		printf("VBO VboImaginnaryID generating failed\n");
		return -1;
	}
	glBindBuffer(GL_ARRAY_BUFFER, VboImaginnaryID);
	if (DataImaginary)
	{
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * DataElemCount, DataImaginary, GL_STATIC_DRAW);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);		
	}
	else
	{
		printf("DataImaginary is not ready\n");
		return -1;
	}

	return 1;
}

static int
Recompute(void)
{
	if(!ComputeKernel)
		return CL_SUCCESS;

	void *values[2];
	size_t sizes[2];

	int err = 0;
	unsigned int v = 0, s = 0, a = 0;
	values[v++] = &ComputeInputOutputReal;
	values[v++] = &ComputeInputOutputImaginary;

	sizes[s++] = sizeof(cl_mem);
	sizes[s++] = sizeof(cl_mem);

	if(Animated || Update)
	{

		glFinish();

		// If use shared context, then data for ComputeInputOutput* is already in Vbo*
#if (USE_GL_ATTACHMENTS)

		err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &ComputeInputOutputReal, 0, 0, 0);
		if (err != CL_SUCCESS)
		{
			printf("Failed to acquire GL object! %d\n", err);
			return EXIT_FAILURE;
		}

		err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &ComputeInputOutputImaginary, 0, 0, 0);
		if (err != CL_SUCCESS)
		{
			printf("Failed to acquire GL object! %d\n", err);
			return EXIT_FAILURE;
		}

#if (DEBUG_INFO)

		float *DataRealReadBack = (float *)calloc(1, sizeof(float) * DataElemCount);
		float *DataImaginaryReadBack = (float *)calloc(1, sizeof(float) * DataElemCount);

		err = clEnqueueReadBuffer( ComputeCommands, ComputeInputOutputReal, CL_TRUE, 0, DataElemCount * sizeof(float), DataRealReadBack, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		err = clEnqueueReadBuffer( ComputeCommands, ComputeInputOutputImaginary, CL_TRUE, 0, DataElemCount * sizeof(float), DataImaginaryReadBack, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		for (int i = 0; i < DataElemCount; ++i)
			printf("Before NDRange %d - %d: [%f %f] - [%f %f]\n", NDRangeCount, i, DataReal[i], DataImaginary[i], DataRealReadBack[i], DataImaginaryReadBack[i]);

		free(DataRealReadBack);
		free(DataImaginaryReadBack);

#endif

#else

		// Not sharing context with OpenGL, needs to explicitly copy/write to exchange data
		err = clEnqueueWriteBuffer(ComputeCommands, ComputeInputOutputReal, 1, 0, 
			DataElemCount * sizeof(float), DataReal, 0, 0, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Failed to write buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		err = clEnqueueWriteBuffer(ComputeCommands, ComputeInputOutputImaginary, 1, 0, 
			DataElemCount * sizeof(float), DataImaginary, 0, 0, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Failed to write buffer! %d\n", err);
			return EXIT_FAILURE;
		}

#endif
		Update = 0;
		err = CL_SUCCESS;
		for (a = 0; a < s; a++)
			err |= clSetKernelArg(ComputeKernel, a, sizes[a], values[a]);

		if (err)
			return -10;

		size_t global[1];
		size_t local[1];

		global[0] = DataElemCount;
		local[0] = 64;

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

#if DEBUG_INFO

		float *DataRealReadBack = (float *)calloc(1, sizeof(float) * DataElemCount);
		float *DataImaginaryReadBack = (float *)calloc(1, sizeof(float) * DataElemCount);

		err = clEnqueueReadBuffer( ComputeCommands, ComputeInputOutputReal, CL_TRUE, 0, DataElemCount * sizeof(float), DataRealReadBack, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		err = clEnqueueReadBuffer( ComputeCommands, ComputeInputOutputImaginary, CL_TRUE, 0, DataElemCount * sizeof(float), DataImaginaryReadBack, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		for (int i = 0; i < DataElemCount; ++i)
			printf("After NDRange %d - %d: [%f %f] - [%f %f]\n", NDRangeCount, i, DataReal[i], DataImaginary[i], DataRealReadBack[i], DataImaginaryReadBack[i]);

		free(DataRealReadBack);
		free(DataImaginaryReadBack);

#endif

#if (USE_GL_ATTACHMENTS)

		// Release control and the data is already in VBOs
		err = clEnqueueReleaseGLObjects(ComputeCommands, 1, &ComputeInputOutputReal, 0, 0, 0);
		if (err != CL_SUCCESS)
		{
			printf("Failed to release GL object! %d\n", err);
			return EXIT_FAILURE;
		}

		err = clEnqueueReleaseGLObjects(ComputeCommands, 1, &ComputeInputOutputImaginary, 0, 0, 0);
		if (err != CL_SUCCESS)
		{
			printf("Failed to release GL object! %d\n", err);
			return EXIT_FAILURE;
		}

#else
		// Explicitly copy data back to host and update VBOs
		err = clEnqueueReadBuffer( ComputeCommands, ComputeInputOutputReal, CL_TRUE, 0, DataElemCount * sizeof(float), DataReal, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		err = clEnqueueReadBuffer( ComputeCommands, ComputeInputOutputImaginary, CL_TRUE, 0, DataElemCount * sizeof(float), DataImaginary, 0, NULL, NULL );      
		if (err != CL_SUCCESS)
		{
			printf("Failed to read buffer! %d\n", err);
			return EXIT_FAILURE;
		}

		UpdateVBOs();
#endif

		clFinish(ComputeCommands);
	}

	return CL_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////

static int 
CreateComputeResource(void)
{
	int err = 0;

#if (USE_GL_ATTACHMENTS)

	if(ComputeInputOutputReal)
		clReleaseMemObject(ComputeInputOutputReal);
	ComputeInputOutputReal = 0;

	if (VboRealID)
	{
		printf("Allocating compute input/output real part for FFT in device memory...\n");
		ComputeInputOutputReal = clCreateFromGLBuffer(ComputeContext, CL_MEM_READ_WRITE, VboRealID, &err);
		if (!ComputeInputOutputReal || err != CL_SUCCESS)
		{
			printf("Failed to create OpenGL VBO reference! %d\n", err);
			return -1;
		}
	}
	else
	{
		printf("VboRealID not valid!\n");
		return -1;
	}

	if (VboImaginnaryID)
	{
		if(ComputeInputOutputImaginary)
			clReleaseMemObject(ComputeInputOutputImaginary);
		ComputeInputOutputImaginary = 0;

		printf("Allocating compute input/output imaginary part for FFT in device memory...\n");
		ComputeInputOutputImaginary = clCreateFromGLBuffer(ComputeContext, CL_MEM_READ_WRITE, VboImaginnaryID, &err);
		if (!ComputeInputOutputImaginary || err != CL_SUCCESS)
		{
			printf("Failed to create OpenGL VBO reference! %d\n", err);
			return -1;
		}
	}
	else
	{
		printf("VboImaginnaryID not valid!\n");
		return -1;
	}

#else

	if(ComputeInputOutputReal)
		clReleaseMemObject(ComputeInputOutputReal);
	ComputeInputOutputReal = 0;

	printf("Allocating compute input/output real part for FFT in device memory...\n");
	ComputeInputOutputReal = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
			sizeof(float) * DataElemCount, 0, &err);
	if (!ComputeInputOutputReal || err != CL_SUCCESS)
	{
		printf("Failed to create OpenGL VBO reference! %d\n", err);
		return -1;
	}

	if(ComputeInputOutputImaginary)
		clReleaseMemObject(ComputeInputOutputImaginary);
	ComputeInputOutputImaginary = 0;

	printf("Allocating compute input/output real part for FFT in device memory...\n");
	ComputeInputOutputImaginary = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
			sizeof(float) * DataElemCount, 0, &err);
	if (!ComputeInputOutputImaginary || err != CL_SUCCESS)
	{
		printf("Failed to create OpenGL VBO reference! %d\n", err);
		return -1;
	}

#endif

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

	return CL_SUCCESS;
}

static void
Cleanup(void)
{
	clFinish(ComputeCommands);
	clReleaseKernel(ComputeKernel);
	clReleaseProgram(ComputeProgram);
	clReleaseCommandQueue(ComputeCommands);
	clReleaseMemObject(ComputeInputOutputReal);
	clReleaseMemObject(ComputeInputOutputImaginary);
	clReleaseContext(ComputeContext);

	ComputeCommands = 0;
	ComputeKernel = 0;
	ComputeProgram = 0;    
	ComputeInputOutputReal = 0;
	ComputeInputOutputImaginary = 0;
	ComputeContext = 0;

	free(DataReal);
	free(DataImaginary);
}

static void
Shutdown(void)
{
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
	glViewport(0, 0, Width, Height);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(2, GL_FLOAT, 0, VertexPos);
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

	err = SetupGLProgram();
	if (err != 1)
	{
		printf ("Failed to setup OpenGL Shader! Error %d\n", err);
		exit (err);
	}

	err = InitData();
	if (err != 1)
	{
		printf ("Failed to Init FFT Data! Error %d\n", err);
		exit (err);
	}

	err = CreateGLResouce();
	if (err != 1)
	{
		printf ("Failed to create GL resource! Error %d\n", err);
		exit (err);
	}

	glFinish();

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

	clFlush(ComputeCommands);

	return CL_SUCCESS;
}


static void
ReportInfo(void)
{
	if(ShowStats)
	{
		int iX = 20;
		int iY = 20;

		DrawText(iX - 1, Height - iY - 1, 0, StatsString);
		DrawText(iX - 2, Height - iY - 2, 0, StatsString);
		DrawText(iX, Height - iY, 1, StatsString);
	}

	if(ShowInfo)
	{
		int iX = TextOffset[0];
		int iY = Height - TextOffset[1];

		DrawText(Width - iX - 1 - strlen(InfoString) * 10, Height - iY - 1, 0, InfoString);
		DrawText(Width - iX - 2 - strlen(InfoString) * 10, Height - iY - 2, 0, InfoString);
		DrawText(Width - iX - strlen(InfoString) * 10, Height - iY, 1, InfoString);

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
	glClear (GL_COLOR_BUFFER_BIT);

	if(Animated)
	{
		UpdateData();
		UpdateVBOs();
	}

	int err = Recompute();
	if (err != 0)
	{
		printf("Error %d from Recompute!\n", err);
		exit(1);
	}

	glDrawArrays(GL_POINTS, 0, DataElemCount);
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

	if(w > 2 * Width || h > 2 * Height)
	{
		Width = w;
		Height = h;
		Cleanup();
		if(Initialize(ComputeDeviceType == CL_DEVICE_TYPE_GPU) != GL_NO_ERROR)
			Shutdown();
	}

	Width = w;
	Height = h;    
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

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (Width, Height);
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

