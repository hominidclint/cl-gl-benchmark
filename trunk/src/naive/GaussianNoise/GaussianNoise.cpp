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

#include "CLUtil.hpp"
#include "SDKBitMap.hpp"
using namespace appsdk;

////////////////////////////////////////////////////////////////////////////////

#define USE_GL_ATTACHMENTS              (0)  // enable OpenGL attachments for Compute results
#define DEBUG_INFO                      (1)     
#define COMPUTE_KERNEL_FILENAME_1       ("GaussianNoiseGL_Kernels.cl")
#define COMPUTE_KERNEL_FILENAME_2       ("GaussianNoiseGL_Kernels2.cl")
#define COMPUTE_KERNEL_METHOD_NAME      ("gaussian_transform")
#define SEPARATOR                       ("----------------------------------------------------------------------\n")

////////////////////////////////////////////////////////////////////////////////

#define INPUT_IMAGE                     ("GaussianNoiseGL_Input.bmp")
#define OUTPUT_IMAGE                    ("GaussianNoiseGL_Output.bmp")

#define GROUP_SIZE                      (64)
#define FACTOR                          (60)

static SDKBitMap                        InputBitmap;
static cl_uchar4                        *InputImageData;
static cl_uchar4                        *OutputImageData;
static uchar4                           *PixelData;
static cl_uint                          PixelSize = sizeof(uchar4);

////////////////////////////////////////////////////////////////////////////////

static cl_context                       ComputeContext;
static cl_command_queue                 ComputeCommands;
static cl_kernel                        ComputeKernel;
static cl_program                       ComputeProgram;
static cl_program                       ComputeProgram1;
static cl_program                       ComputeProgram2;
static cl_device_id                     ComputeDeviceId;
static cl_device_type                   ComputeDeviceType;
static cl_mem                           ComputeInputImage;
static cl_mem                           ComputeOutputImage;
static size_t                           MaxBlockSize;
static size_t                           BlockSize[2];
static unsigned int                     NDRangeCount = 0;

////////////////////////////////////////////////////////////////////////////////

static int VarFactor                    = FACTOR;
static int Incre                        = 2;

////////////////////////////////////////////////////////////////////////////////

static int Width                        = 0;
static int Height                       = 0;

static int Animated                     = 0;
static int Update                       = 1;

////////////////////////////////////////////////////////////////////////////////

static uint TextureId                   = 0;
static uint TextureTarget               = GL_TEXTURE_2D;
static uint TextureInternal             = GL_RGBA;
static uint TextureFormat               = GL_RGBA;
static uint TextureType                 = GL_UNSIGNED_BYTE;
static uint TextureWidth                = 0;
static uint TextureHeight               = 0;
static uint ActiveTextureUnit           = GL_TEXTURE1_ARB;

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

static int EnableOutput                 = 0;
FILE *fp;

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

static int ReadInputImage(const char *file_name)
{
	InputBitmap.load(file_name);
	if(!InputBitmap.isLoaded())
	{
		printf("Failed to load input image!\n");
		return -1;
	}

	return 1;
}

static int WriteOutputImage(const char *file_name)
{
    // copy output image data back to original pixel data
    memcpy(PixelData, OutputImageData, Width * Height * PixelSize);

    // write the output bmp file
    if(!InputBitmap.write(file_name))
    {
        error("Failed to write output image!");
        return -1;
    }

    return 1;

}
static int InitData()
{
	int err;

	err = ReadInputImage(INPUT_IMAGE);
	if (err != 1)
		return -1;

    // Get width and height of input image
	Width = InputBitmap.getWidth();
	Height = InputBitmap.getHeight();
	TextureWidth = Width;
	TextureHeight = Height;

    // Allocate memory for input & output image data
	if (InputImageData)
		free(InputImageData);
	InputImageData = NULL;

	InputImageData = (cl_uchar4 *)calloc(1, Width * Height * sizeof(cl_uchar4));
	if (!InputImageData)
	{
		printf("Failed to allocate memory (InputImageData)\n");
		return -1;
	}

	if (OutputImageData)
		free(OutputImageData);
	OutputImageData = NULL;

	OutputImageData = (cl_uchar4 *)calloc(1, Width * Height * sizeof(cl_uchar4));
	if (!OutputImageData)
	{
		printf("Failed to allocate memory (OutputImageData)\n");
		return -1;
	}

    // get the pointer to pixel data
	PixelData = InputBitmap.getPixels();
	if(PixelData == NULL)
	{
		printf("Failed to read pixel Data!\n");
		return -1;
	}

    // Copy pixel data into InputImageData
	memcpy(InputImageData, PixelData, Width * Height * PixelSize);


#if (DEBUG_INFO)
    // copy output image data back to original pixel data
    memcpy(PixelData, InputImageData, Width * Height * PixelSize);

    // write the output bmp file
    if(!InputBitmap.write("AfterInputRead.bmp"))
    {
        error("Failed to write output image!");
        return -1;
    }
#endif

	return CL_SUCCESS;
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
	glEnable(TextureTarget);	
	glGenTextures(1, &TextureId);
	glBindTexture(TextureTarget, TextureId);
	glTexParameteri(TextureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(TextureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(TextureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(TextureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(TextureTarget, 0, TextureInternal, TextureWidth, TextureHeight, 0, 
		TextureFormat, TextureType, 0);
	glBindTexture(TextureTarget, 0);
}

static void 
RenderTexture( void *pvData )
{
	glClearColor (0.0, 0.0, 0.0, 0.0);
	glClear (GL_COLOR_BUFFER_BIT);

	glEnable( TextureTarget );
	glBindTexture( TextureTarget, TextureId );

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, Width, Height);

#if (USE_GL_ATTACHMENTS)
	// Already in GPU memory, just need to render
#else
	// Need to copy to texture
	if(pvData)
		glTexSubImage2D(TextureTarget, 0, 0, 0, TextureWidth, TextureHeight, 
			TextureFormat, TextureType, pvData);
#endif

	glTexParameteri(TextureTarget, GL_TEXTURE_COMPARE_MODE_ARB, GL_NONE);
	glBegin( GL_QUADS );
	{
		glColor3f(1.0f, 1.0f, 1.0f);
		glTexCoord2f( 0.0f, 0.0f );
		glVertex3f( -1.0f, -1.0f, 0.5f );

		glTexCoord2f( 0.0f, 1.0f );
		glVertex3f( -1.0f, 1.0f, 0.5f );

		glTexCoord2f( 1.0f, 1.0f );
		glVertex3f( 1.0f, 1.0f, 0.5f );

		glTexCoord2f( 1.0f, 0.0f );
		glVertex3f( 1.0f, -1.0f, 0.5f );
	}
	glEnd();
	glBindTexture( TextureTarget, 0 );
	glDisable( TextureTarget );
}

static int
Recompute(void)
{
	glFinish();

	if(!ComputeKernel || !ComputeOutputImage)
		return CL_SUCCESS;

	int err = 0;

#if (USE_GL_ATTACHMENTS)

	// Get control from GL context
	err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &ComputeOutputImage, 0, 0, 0);
	if (err != CL_SUCCESS)
	{
		printf("%s: Failed to acquire GL object! %d\n", __FUNCTION__, err);
		return EXIT_FAILURE;
	}

#else

	// Nothing needs to be done here

#endif

	void *values[3];
	size_t sizes[3];

	unsigned int v = 0, s = 0, a = 0;
	values[v++] = &ComputeInputImage;
	values[v++] = &ComputeOutputImage;
	values[v++] = &VarFactor;

	sizes[s++] = sizeof(cl_mem);
	sizes[s++] = sizeof(cl_mem);
	sizes[s++] = sizeof(int);

	if(Animated || Update)
	{
		Update = 0;
		err = CL_SUCCESS;
		for (a = 0; a < s; a++)
			err |= clSetKernelArg(ComputeKernel, a, sizes[a], values[a]);

		if (err)
			return -10;
	}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if (DEBUG_INFO)
	glFinish();

	cl_uchar4 *InputDataReadBack = (cl_uchar4 *)calloc(1, PixelSize * TextureWidth *  TextureHeight);
	cl_uchar4 *OutputDataReadBack = (cl_uchar4 *)calloc(1, PixelSize * TextureWidth *  TextureHeight);

	err = clEnqueueReadBuffer( ComputeCommands, ComputeInputImage, CL_TRUE, 0, PixelSize * TextureWidth *  TextureHeight, InputDataReadBack, 0, NULL, NULL);      
	if (err != CL_SUCCESS)
	{
		printf("%s: Failed to read input image buffer! %d\n", __FUNCTION__, err);
		return EXIT_FAILURE;
	}

	err = clEnqueueReadBuffer(ComputeCommands, ComputeOutputImage, CL_TRUE, 0, PixelSize * TextureWidth *  TextureHeight, OutputDataReadBack, 0, NULL, NULL);      
	if (err != CL_SUCCESS)
	{
		printf("%s: Failed to read output image buffer! %d\n", __FUNCTION__, err);
		return EXIT_FAILURE;
	}

    // copy output image data back to original pixel data
    memcpy(PixelData, InputDataReadBack, Width * Height * PixelSize);

    // write the output bmp file
    if(!InputBitmap.write("InputBeforeNDRange.bmp"))
    {
        printf("%s: Failed to write output image!\n", __FUNCTION__);
        return -1;
    }

    // copy output image data back to original pixel data
    memcpy(PixelData, OutputDataReadBack, Width * Height * PixelSize);

    // write the output bmp file
    if(!InputBitmap.write("OutputBeforeNDRange.bmp"))
    {
        printf("%s: Failed to write output image!", __FUNCTION__);
        return -1;
    }

	free(InputDataReadBack);
	free(OutputDataReadBack);
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    size_t globalThreads[] = {Width/2, Height};
    size_t localThreads[] = {BlockSize[0], BlockSize[1]};

#if (DEBUG_INFO)
	if(FrameCount <= 1)
		printf("Global[%4d %4d] Local[%4d %4d]\n", 
			(int)globalThreads[0], (int)globalThreads[1],
			(int)localThreads[0], (int)localThreads[1]);
#endif

	err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel, 2, NULL, globalThreads, localThreads, 0, NULL, NULL);
	if (err)
	{
		printf("Failed to enqueue kernel! %d\n", err);
		return err;
	}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if (DEBUG_INFO)
	glFinish();

	cl_uchar4 *InputDataReadBack1 = (cl_uchar4 *)calloc(1, PixelSize * TextureWidth *  TextureHeight);
	cl_uchar4 *OutputDataReadBack2 = (cl_uchar4 *)calloc(1, PixelSize * TextureWidth *  TextureHeight);

	err = clEnqueueReadBuffer( ComputeCommands, ComputeInputImage, CL_TRUE, 0, PixelSize * TextureWidth *  TextureHeight, InputDataReadBack1, 0, NULL, NULL);      
	if (err != CL_SUCCESS)
	{
		printf("%s: Failed to read input image buffer! %d\n", __FUNCTION__, err);
		return EXIT_FAILURE;
	}

	err = clEnqueueReadBuffer(ComputeCommands, ComputeOutputImage, CL_TRUE, 0, PixelSize * TextureWidth *  TextureHeight, OutputDataReadBack2, 0, NULL, NULL);      
	if (err != CL_SUCCESS)
	{
		printf("%s: Failed to read output image buffer! %d\n", __FUNCTION__, err);
		return EXIT_FAILURE;
	}

    // copy output image data back to original pixel data
    memcpy(PixelData, InputDataReadBack1, Width * Height * PixelSize);

    // write the output bmp file
    if(!InputBitmap.write("InputAfterNDRange.bmp"))
    {
        printf("%s: Failed to write output image!\n", __FUNCTION__);
        return -1;
    }

    // copy output image data back to original pixel data
    memcpy(PixelData, OutputDataReadBack2, Width * Height * PixelSize);

    // write the output bmp file
    if(!InputBitmap.write("OutputAfterNDRange.bmp"))
    {
        printf("%s: Failed to write output image!", __FUNCTION__);
        return -1;
    }

	free(InputDataReadBack1);
	free(OutputDataReadBack2);
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if (USE_GL_ATTACHMENTS)

	// Return control to GL context, data already in texture object
	err = clEnqueueReleaseGLObjects(ComputeCommands, 1, &ComputeOutputImage, 0, 0, 0);
	if (err != CL_SUCCESS)
	{
		printf("Failed to release GL object! %d\n", err);
		return EXIT_FAILURE;
	}

#else
	// Need to explicitly copy to host side for later rendering
	err = clEnqueueReadBuffer( ComputeCommands, ComputeOutputImage, CL_TRUE, 0, PixelSize * TextureWidth *  TextureHeight, OutputImageData, 0, NULL, NULL);      
	if (err != CL_SUCCESS)
	{
		printf("Failed to read image! %d\n", err);
		return EXIT_FAILURE;
	}

#endif	

	clFinish(ComputeCommands);

	NDRangeCount++;

	return CL_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////

static int 
CreateComputeBuffers(void)
{
	int err = 0;

#if (USE_GL_ATTACHMENTS)

	if(ComputeOutputImage)
		clReleaseMemObject(ComputeOutputImage);
	ComputeOutputImage = 0;

	printf("Allocating compute result image in device memory...\n");
	ComputeOutputImage = clCreateFromGLTexture(ComputeContext, CL_MEM_READ_WRITE, TextureTarget, 0, TextureId, &err);
	if (!ComputeOutputImage || err != CL_SUCCESS)
	{
		printf("Failed to create OpenGL texture reference! %d\n", err);
		return -1;
	}

#else

	if(ComputeOutputImage)
		clReleaseMemObject(ComputeOutputImage);
	ComputeOutputImage = 0;

	printf("Allocating compute output image in device memory...\n");
	ComputeOutputImage = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, PixelSize * TextureWidth * TextureHeight, NULL, &err);
	if (!ComputeOutputImage || err != CL_SUCCESS)
	{
		printf("Failed to create OpenGL output buffer! %d\n", err);
		return -1;
	}

	if (OutputImageData)
		free(OutputImageData);

	printf("Allocating compute output image in host memory...\n");
	OutputImageData = (cl_uchar4 *)calloc(1, TextureWidth * TextureHeight * PixelSize);
	if(!OutputImageData)
	{
		printf("Failed to create host image buffer!\n");
		return -1;
	}

#endif

	if(ComputeInputImage)
		clReleaseMemObject(ComputeInputImage);
	ComputeInputImage = 0;

	printf("Allocating compute input image in host memory...\n");
	ComputeInputImage = clCreateBuffer(ComputeContext, CL_MEM_READ_ONLY, PixelSize * TextureWidth * TextureHeight, NULL, &err);
	if (!ComputeInputImage || err != CL_SUCCESS)
	{
		printf("Failed to create OpenCL input buffer!\n");
		return -1;
	}

	printf("Sending data to input image buffer in device memory...\n");
	err = clEnqueueWriteBuffer(ComputeCommands, ComputeInputImage, CL_FALSE, 0, PixelSize * TextureWidth * TextureHeight, InputImageData, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Failed to send data to input buffer\n");
		return -1;
	}

#if (DEBUG_INFO)
	glFinish();

#if (USE_GL_ATTACHMENTS)
	// Get control from GL context
	err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &ComputeOutputImage, 0, 0, 0);
	if (err != CL_SUCCESS)
	{
		printf("Failed to acquire GL object! %d\n", err);
		return EXIT_FAILURE;
	}
#endif

	cl_uchar4 *InputDataReadBack = (cl_uchar4 *)calloc(1, PixelSize * TextureWidth *  TextureHeight);
	cl_uchar4 *OutputDataReadBack = (cl_uchar4 *)calloc(1, PixelSize * TextureWidth *  TextureHeight);

	err = clEnqueueReadBuffer( ComputeCommands, ComputeInputImage, CL_TRUE, 0, PixelSize * TextureWidth *  TextureHeight, InputDataReadBack, 0, NULL, NULL);      
	if (err != CL_SUCCESS)
	{
		printf("%s: Failed to read input image buffer! %d\n", __FUNCTION__, err);
		return EXIT_FAILURE;
	}

	err = clEnqueueReadBuffer(ComputeCommands, ComputeOutputImage, CL_TRUE, 0, PixelSize * TextureWidth *  TextureHeight, OutputDataReadBack, 0, NULL, NULL);      
	if (err != CL_SUCCESS)
	{
		printf("%s: Failed to read output image buffer! %d\n", __FUNCTION__, err);
		return EXIT_FAILURE;
	}

    // copy output image data back to original pixel data
    memcpy(PixelData, InputDataReadBack, Width * Height * PixelSize);

    // write the output bmp file
    if(!InputBitmap.write("InputAfterCreateBuffer.bmp"))
    {
        printf("%s: Failed to write output image!\n", __FUNCTION__);
        return -1;
    }

    // copy output image data back to original pixel data
    memcpy(PixelData, OutputDataReadBack, Width * Height * PixelSize);

    // write the output bmp file
    if(!InputBitmap.write("OutputAfterCreateBuffer.bmp"))
    {
        printf("%s: Failed to write output image!", __FUNCTION__);
        return -1;
    }

	free(InputDataReadBack);
	free(OutputDataReadBack);

#if (USE_GL_ATTACHMENTS)
	// Return control to GL context, data already in texture
	err = clEnqueueReleaseGLObjects(ComputeCommands, 1, &ComputeOutputImage, 0, 0, 0);
	if (err != CL_SUCCESS)
	{
		printf("Failed to release GL object! %d\n", err);
		return EXIT_FAILURE;
	}
#endif

	clFinish(ComputeCommands);
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
SetupComputeKernel(void)
{
	int err = 0;
	char *source = 0;
	size_t length = 0;

	if(ComputeKernel)
		clReleaseKernel(ComputeKernel);    
	ComputeKernel = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Program 1
	if(ComputeProgram1)
		clReleaseProgram(ComputeProgram1);
	ComputeProgram1 = 0;

	printf(SEPARATOR);
	printf("Loading kernel source from file '%s'...\n", COMPUTE_KERNEL_FILENAME_1);    
	err = LoadTextFromFile(COMPUTE_KERNEL_FILENAME_1, &source, &length);
	if (!source || err)
	{
		printf("Error: Failed to load kernel source 1!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source buffer
	//
	ComputeProgram1 = clCreateProgramWithSource(ComputeContext, 1, (const char**)&source, NULL, &err);
	if (!ComputeProgram1 || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute program 1!\n");
		return EXIT_FAILURE;
	}
	free(source);

	err = clCompileProgram(ComputeProgram1, 0, 0, 0, 0, 0, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to compile compute program 1!\n");
		return EXIT_FAILURE;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Program 2
	if(ComputeProgram2)
		clReleaseProgram(ComputeProgram2);
	ComputeProgram2 = 0;

	printf(SEPARATOR);
	printf("Loading kernel source from file '%s'...\n", COMPUTE_KERNEL_FILENAME_2);    
	err = LoadTextFromFile(COMPUTE_KERNEL_FILENAME_2, &source, &length);
	if (!source || err)
	{
		printf("Error: Failed to load kernel source 2!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source buffer
	//
	ComputeProgram2 = clCreateProgramWithSource(ComputeContext, 1, (const char**)&source, NULL, &err);
	if (!ComputeProgram2 || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute program 2!\n");
		return EXIT_FAILURE;
	}
	free(source);

	err = clCompileProgram(ComputeProgram2, 0, 0, 0, 0, 0, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to compile compute program 2!\n");
		return EXIT_FAILURE;
	}

	cl_program Programs[] = { ComputeProgram1, ComputeProgram2};
	ComputeProgram = clLinkProgram(ComputeContext,
									0,
									0,
									0,
									2,
									Programs,
									NULL,
									NULL,
									&err
									);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to link compute programs!\n");
		return EXIT_FAILURE;
	}

    // Create the compute kernel from within the program
    //
	printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME);    
	ComputeKernel = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME, &err);
	if (!ComputeKernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		return EXIT_FAILURE;
	}

    // Get the maximum work group size for executing the kernel on the device
    //
	err = clGetKernelWorkGroupInfo(ComputeKernel, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &MaxBlockSize, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to retrieve kernel work group info! %d\n", err);
		exit(1);
	}

#if (DEBUG_INFO)
	printf("MaxBlockSize: %d\n", MaxBlockSize);
#endif

	BlockSize[0] = GROUP_SIZE;
	BlockSize[1] = 1;

	if (BlockSize[0] * BlockSize[1] > MaxBlockSize)
	{
		BlockSize[0] = MaxBlockSize;
		BlockSize[1] = 1;
	}

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
	clReleaseMemObject(ComputeOutputImage);
	clReleaseMemObject(ComputeInputImage);
	clReleaseContext(ComputeContext);

	ComputeCommands = 0;
	ComputeKernel = 0;
	ComputeProgram = 0;    
	ComputeOutputImage = 0;
	ComputeInputImage = 0;
	ComputeContext = 0;
}

static void
Shutdown(void)
{
	if (EnableOutput)
		fclose(fp);
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
		printf("Qjulia requires images: Images not supported on this device.");
		return CL_IMAGE_FORMAT_NOT_SUPPORTED;
	}

	err = SetupComputeKernel();
	if (err != CL_SUCCESS)
	{
		printf ("Failed to setup compute kernel! Error %d\n", err);
		exit (err);
	}

	err = CreateComputeBuffers();
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
		if (EnableOutput)
			fprintf(fp, "%s\n", StatsString);
		FrameCount = 0;
		TimeElapsed = 0;
	}    
}

static void
Display_(void)
{
	FrameCount++;
	uint64_t uiStartTime = GetCurrentTime();

	if(Animated)
	{
		if (VarFactor == 100)
			Incre = -2;
		else if (VarFactor == 20)
			Incre = 2;
		VarFactor += Incre;
	}

	int err = Recompute();
	if (err != 0)
	{
		printf("Error %d from Recompute!\n", err);
		exit(1);
	}

	RenderTexture(OutputImageData);
	ReportInfo();

    glFinish(); // for timing
    
    uint64_t uiEndTime = GetCurrentTime();
    ReportStats(uiStartTime, uiEndTime);
    DrawText(TextOffset[0], TextOffset[1], 1, (Animated == 0) ? "Press space to animate" : " ");
    WriteOutputImage(OUTPUT_IMAGE);
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

		case '+':
		VarFactor += 2;
		break;

		case '-':
		VarFactor -= 2;
		break;

		case 'f':
		glutFullScreen(); 
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
	int err;
	int use_gpu = 1;
	for( i = 0; i < argc && argv; i++)
	{
		if(!argv[i])
			continue;

        if(strstr(argv[i], "-cpu"))
            use_gpu = 0;        

        else if(strstr(argv[i], "-gpu"))
            use_gpu = 1;

        else if(strstr(argv[i], "-animate"))
            Animated = 1;

		else if(strstr(argv[i], "-output"))
		{
			EnableOutput = 1;
			fp = fopen(argv[i+1], "w+");
		}
	}

	err = InitData();
	if (err != CL_SUCCESS)
	{
		printf("Fail to init data\n");
		exit(err);
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

