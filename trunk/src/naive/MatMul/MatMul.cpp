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

#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/freeglut.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>

////////////////////////////////////////////////////////////////////////////////

#define USE_GL_ATTACHMENTS              (0)  // enable OpenGL attachments for Compute results
#define DEBUG_INFO                      (0)     
#define COMPUTE_KERNEL_FILENAME         ("MatrixMultiplication_Kernels.cl")
#define COMPUTE_KERNEL_MATMUL_NAME      ("mmmKernel")
#define COMPUTE_KERNEL_MATMUL_LDS_NAME  ("mmmKernel_local")
#define SEPARATOR                       ("----------------------------------------------------------------------\n")
#define WIDTH                           (512)
#define HEIGHT                          (512)

////////////////////////////////////////////////////////////////////////////////

static cl_context                       ComputeContext;
static cl_command_queue                 ComputeCommands;
static cl_kernel                        ComputeKernel;
static cl_program                       ComputeProgram;
static cl_device_id                     ComputeDeviceId;
static cl_device_type                   ComputeDeviceType;
static cl_mem                           ComputeMatrixA;
static cl_mem                           ComputeMatrixB;
static cl_mem                           ComputeMatrixC;
static cl_mem                           ComputeImage;
static size_t                           MaxWorkGroupSize;
static int                              WorkGroupSize[2];
static int                              WorkGroupItems = 32;

////////////////////////////////////////////////////////////////////////////////

static int Width0                       = 512;
static int Height0                      = 512;
static int Width1                       = 512;
static int Height1                      = 512;

static int Width                        = Width1;
static int Height                       = Height0;

static int Animated                     = 0;
static int Update                       = 1;
static int Lds                          = 0;

static float *Input0                    = NULL;
static float *Input1                    = NULL;
static float *Output                    = NULL;

static int BlockSize                    = 8;

////////////////////////////////////////////////////////////////////////////////

static uint TextureId                   = 0;
static uint TextureTarget               = GL_TEXTURE_2D;
static uint TextureInternal             = GL_RGBA;
static uint TextureFormat               = GL_RGBA;
static uint TextureType                 = GL_UNSIGNED_BYTE;
static uint TextureWidth                = WIDTH;
static uint TextureHeight               = HEIGHT;
static size_t TextureTypeSize           = sizeof(char);
static uint ActiveTextureUnit           = GL_TEXTURE1_ARB;
static void* HostImageBuffer            = 0;

static double TimeElapsed               = 0;
static int FrameCount                   = 0;
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
static float TexCoords[4][2];

////////////////////////////////////////////////////////////////////////////////

static long
GetCurrentTime()
{
	struct timeval tv;

	gettimeofday(&tv, NULL);

	return tv.tv_sec * 1000 + tv.tv_usec/1000.0;
    // return time(NULL);
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
CreateTexture(uint width, uint height)
{    
	if(TextureId)
		glDeleteTextures(1, &TextureId);
	TextureId = 0;

	printf("Creating Texture %d x %d...\n", width, height);

	TextureWidth = width;
	TextureHeight = height;

	glActiveTextureARB(ActiveTextureUnit);
	glGenTextures(1, &TextureId);
	glBindTexture(TextureTarget, TextureId);
	glTexParameteri(TextureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(TextureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(TextureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(TextureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(TextureTarget, 0, TextureInternal, TextureWidth, TextureHeight, 0, 
		TextureFormat, TextureType, 0);
	glBindTexture(TextureTarget, 0);
}

static void 
RenderTexture( void *pvData )
{
	glDisable( GL_LIGHTING );

	glViewport( 0, 0, Width, Height );

	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluOrtho2D( -1.0, 1.0, -1.0, 1.0 );

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	glMatrixMode( GL_TEXTURE );
	glLoadIdentity();

	glEnable( TextureTarget );
	glBindTexture( TextureTarget, TextureId );

	if(pvData)
		glTexSubImage2D(TextureTarget, 0, 0, 0, TextureWidth, TextureHeight, 
			TextureFormat, TextureType, pvData);

	glTexParameteri(TextureTarget, GL_TEXTURE_COMPARE_MODE_ARB, GL_NONE);
	glBegin( GL_QUADS );
	{
		glColor3f(1.0f, 1.0f, 1.0f);
		glTexCoord2f( 0.0f, 0.0f );
		glVertex3f( -1.0f, -1.0f, 0.0f );

		glTexCoord2f( 0.0f, 1.0f );
		glVertex3f( -1.0f, 1.0f, 0.0f );

		glTexCoord2f( 1.0f, 1.0f );
		glVertex3f( 1.0f, 1.0f, 0.0f );

		glTexCoord2f( 1.0f, 0.0f );
		glVertex3f( 1.0f, -1.0f, 0.0f );
	}
	glEnd();
	glBindTexture( TextureTarget, 0 );
	glDisable( TextureTarget );
}

static void
RandomFillArray_Float(float *arrayPtr, int width, int height, float rangeMin, float rangeMax)
{
	if (arrayPtr)
	{
		unsigned int seed = (unsigned int)time(NULL);

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
Recompute(void)
{
	if(!ComputeKernel || !ComputeMatrixC)
		return CL_SUCCESS;

	void *values[5];
	size_t sizes[5];
	size_t global[2];
	size_t local[2];

	int err = 0;
	unsigned int v = 0, s = 0, a = 0;
	values[v++] = &ComputeMatrixA;
	values[v++] = &ComputeMatrixB;
	values[v++] = &ComputeMatrixC;
	values[v++] = &Width0;
	if (Lds)
		values[v++] = NULL;
	else
		values[v++] = &Width1;

	sizes[s++] = sizeof(cl_mem);
	sizes[s++] = sizeof(cl_mem);
	sizes[s++] = sizeof(cl_mem);
	sizes[s++] = sizeof(cl_int);
	if (Lds)
		sizes[s++] = (BlockSize * 4) * (BlockSize * 4) * sizeof(cl_float);
	else
		sizes[s++] = sizeof(cl_int);

	if(Animated || Update)
	{
		clEnqueueWriteBuffer(ComputeCommands, ComputeMatrixA, CL_TRUE, 0, Width0 * Height0 * sizeof(float), Input0, 0, NULL, NULL);
		clEnqueueWriteBuffer(ComputeCommands, ComputeMatrixB, CL_TRUE, 0, Width1 * Height1 * sizeof(float), Input1, 0, NULL, NULL);
		clFlush(ComputeCommands);

		Update = 0;
		err = CL_SUCCESS;
		for (a = 0; a < s; a++)
			err |= clSetKernelArg(ComputeKernel, a, sizes[a], values[a]);

		if (err)
			return -10;
	}

	global[0] = Width1 / 4;
	global[1] = Height0/ 4;
	local[0] = BlockSize;
	local[1] = BlockSize;

#if (DEBUG_INFO)
	if(FrameCount <= 1)
		printf("Global[%4d %4d] Local[%4d %4d]\n", 
			(int)global[0], (int)global[1],
			(int)local[0], (int)local[1]);
#endif

	err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel, 2, NULL, global, local, 0, NULL, NULL);
	if (err)
	{
		printf("Failed to enqueue kernel! %d\n", err);
		return err;
	}

#if (USE_GL_ATTACHMENTS)

	err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &ComputeImage, 0, 0, 0);
	if (err != CL_SUCCESS)
	{
		printf("Failed to acquire GL object! %d\n", err);
		return EXIT_FAILURE;
	}

	size_t origin[] = { 0, 0, 0 };
	size_t region[] = { TextureWidth, TextureHeight, 1 };
	err = clEnqueueCopyBufferToImage(ComputeCommands, ComputeMatrixC, ComputeImage, 
		0, origin, region, 0, NULL, 0);

	if(err != CL_SUCCESS)
	{
		printf("Failed to copy buffer to image! %d\n", err);
		return EXIT_FAILURE;
	}

	err = clEnqueueReleaseGLObjects(ComputeCommands, 1, &ComputeImage, 0, 0, 0);
	if (err != CL_SUCCESS)
	{
		printf("Failed to release GL object! %d\n", err);
		return EXIT_FAILURE;
	}

#else

	err = clEnqueueReadBuffer( ComputeCommands, ComputeMatrixC, CL_TRUE, 0, Width * Height * TextureTypeSize * 4, HostImageBuffer, 0, NULL, NULL );      
	if (err != CL_SUCCESS)
	{
		printf("Failed to read buffer! %d\n", err);
		return EXIT_FAILURE;
	}

#endif

	return CL_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////

static int 
CreateComputeResource(void)
{

#if (USE_GL_ATTACHMENTS)

	if(ComputeImage)
		clReleaseMemObject(ComputeImage);
	ComputeImage = 0;

	printf("Allocating compute result image in device memory...\n");
	ComputeImage = clCreateFromGLTexture(ComputeContext, CL_MEM_WRITE_ONLY, TextureTarget, 0, TextureId, &err);
	if (!ComputeImage || err != CL_SUCCESS)
	{
		printf("Failed to create OpenGL texture reference! %d\n", err);
		return -1;
	}

#else

	if (HostImageBuffer)
		free(HostImageBuffer);

	printf("Allocating compute result image in host memory...\n");
	HostImageBuffer = malloc(TextureWidth * TextureHeight * TextureTypeSize * 4);
	if(!HostImageBuffer)
	{
		printf("Failed to create host image buffer!\n");
		return -1;
	}

	memset(HostImageBuffer, 0, TextureWidth * TextureHeight * TextureTypeSize * 4);

#endif

	if(ComputeMatrixA)
		clReleaseMemObject(ComputeMatrixA);
	ComputeMatrixA = 0;

	ComputeMatrixA = clCreateBuffer(ComputeContext, CL_MEM_READ_ONLY, sizeof(cl_float) * Width0 * Height0, NULL, NULL);
	if (!ComputeMatrixA)
	{
		printf("Failed to create OpenCL array!\n");
		return -1;
	}

	if(ComputeMatrixB)
		clReleaseMemObject(ComputeMatrixB);
	ComputeMatrixB = 0;

	ComputeMatrixB = clCreateBuffer(ComputeContext, CL_MEM_READ_ONLY, sizeof(cl_float) * Width1 * Height1, NULL, NULL);
	if (!ComputeMatrixB)
	{
		printf("Failed to create OpenCL array!\n");
		return -1;
	}


	if(ComputeMatrixC)
		clReleaseMemObject(ComputeMatrixC);
	ComputeMatrixC = 0;

	ComputeMatrixC = clCreateBuffer(ComputeContext, CL_MEM_WRITE_ONLY, TextureTypeSize * 4 * TextureWidth * TextureHeight, NULL, NULL);
	if (!ComputeMatrixC)
	{
		printf("Failed to create OpenCL array!\n");
		return -1;
	}

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
	if (Lds)
	{
		printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_MATMUL_LDS_NAME); 
		ComputeKernel = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_MATMUL_LDS_NAME, &err);
	}
	else
	{
		printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_MATMUL_NAME); 
		ComputeKernel = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_MATMUL_NAME, &err);
	}

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
	WorkGroupSize[1] = MaxWorkGroupSize / WorkGroupSize[0];

	printf(SEPARATOR);

	return CL_SUCCESS;

}

static void
Cleanup(void)
{
	clFinish(ComputeCommands);
	clReleaseKernel(ComputeKernel);
	clReleaseProgram(ComputeProgram);
	clReleaseCommandQueue(ComputeCommands);
	clReleaseMemObject(ComputeMatrixA);
	clReleaseMemObject(ComputeMatrixB);
	clReleaseMemObject(ComputeMatrixC);
	clReleaseMemObject(ComputeImage);
	clReleaseContext(ComputeContext);

	ComputeCommands = 0;
	ComputeKernel = 0;
	ComputeProgram = 0;    
	ComputeMatrixA = 0;
	ComputeMatrixB = 0;
	ComputeMatrixC = 0;
	ComputeImage = 0;
	ComputeContext = 0;
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
	CreateTexture(Width, Height);

	glClearColor (0.0, 0.0, 0.0, 0.0);

	glDisable(GL_DEPTH_TEST);
	glActiveTexture(GL_TEXTURE0);
	glViewport(0, 0, Width, Height);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	TexCoords[3][0] = 0.0f;
	TexCoords[3][1] = 0.0f;
	TexCoords[2][0] = Width;
	TexCoords[2][1] = 0.0f;
	TexCoords[1][0] = Width;
	TexCoords[1][1] = Height;
	TexCoords[0][0] = 0.0f;
	TexCoords[0][1] = Height;

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(2, GL_FLOAT, 0, VertexPos);
	glClientActiveTexture(GL_TEXTURE0);
	glTexCoordPointer(2, GL_FLOAT, 0, TexCoords);
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

	cl_bool image_support;
	err = clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_IMAGE_SUPPORT,
		sizeof(image_support), &image_support, NULL);
	if (err != CL_SUCCESS) {
		printf("Unable to query device for image support");
		exit(err);
	}
	if (image_support == CL_FALSE) {
		printf("MatMul requires images: Images not supported on this device.");
		return CL_IMAGE_FORMAT_NOT_SUPPORTED;
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

	Input0 = CreateRandomFilledArray_Float(Width0, Height0, 0.0, 1.0);
	Input1 = CreateRandomFilledArray_Float(Width1, Height1, 0.0, 1.0);
	Output = CreateRandomFilledArray_Float(Width1, Height0, 0.0, 1.0);

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
		RandomFillArray_Float(Input0, Width0, Height0, 0.0, 1.0);
		RandomFillArray_Float(Input1, Width1, Height1, 0.0, 1.0);
	}

	int err = Recompute();
	if (err != 0)
	{
		printf("Error %d from Recompute!\n", err);
		exit(1);
	}

	RenderTexture(HostImageBuffer);
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

		else if (strstr(argv[i], "lds"))
			Lds = 1;
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

