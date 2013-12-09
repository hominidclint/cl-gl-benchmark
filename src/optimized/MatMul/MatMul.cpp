/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include <unistd.h>

#include "MatMul.hpp"

MatrixMultiplication *me;           /**< Pointing to MatMul class */

/// Global variables for GL context 
GLXContext gGlCtxSep;
#define GLX_CONTEXT_MAJOR_VERSION_ARB           0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB           0x2092
typedef GLXContext (*GLXCREATECONTEXTATTRIBSARBPROC)(Display*, GLXFBConfig,
	GLXContext, Bool, const int*);
Window          winSep;
Display         *displayNameSep;
XEvent          xevSep;


/// GL Shader Source
const GLchar *vertex_shader =
"#version 430\n"

"layout(location=0) in vec1 in_Color_0;\n"
"layout(location=1) in vec1 in_Color_1;\n"
"layout(location=2) in vec1 in_Color_2;\n"
"layout(location=3) in vec2 in_Position;\n"
"out vec4 ex_Color;\n"

"void main(void)\n"
"{\n"
" gl_Position = vec4(in_Position.x, in_Position.y, 0.0, 1.0);\n"
" ex_Color = vec4(in_Color_0, in_Color_1, in_Color_2, 1.0);\n"
"}\n";


const GLchar *fragment_shader =
"#version 130\n"

"in vec4 ex_Color;\n"
"out vec4 out_Color;\n"

"void main(void)\n"
"{\n"
" out_Color = ex_Color;\n"
"}\n";

int MatrixMultiplication::clPrepareContext(int argc, char **argv)
{    
    cl_int status = CL_SUCCESS;
    cl_device_type dType;

    if(Args->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //Args->deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(Args->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, Args->platformId,
                               Args->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");

    // Init Context
    clInitContextFromGL(argc, argv);

    // getting device on which to run the sample
    // First, get the size of device list data
    size_t deviceListSize = 0;

    status = clGetContextInfo(
                 context,
                 CL_CONTEXT_DEVICES,
                 0,
                 NULL,
                 &deviceListSize);
    CHECK_OPENCL_ERROR(status, "clGetContextInfo failed.");

    int deviceCount = (int)(deviceListSize / sizeof(cl_device_id));
    deviceCount = deviceCount;

    devices = (cl_device_id *)malloc(deviceListSize);
    CHECK_ALLOCATION((devices), "Failed to allocate memory (devices).");

    // Now, get the device list data
    status = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              deviceListSize,
                              (devices),
                              NULL);
    CHECK_OPENCL_ERROR(status, "clGetGetContextInfo failed.");

    if (dType == CL_DEVICE_TYPE_CPU)
    {
        interopDeviceId = devices[Args->deviceId];
    }

    // Create command queue

    cl_command_queue_properties prop = 0;
    commandQueue = clCreateCommandQueue(
                       context,
                       interopDeviceId,
                       prop,
                       &status);
    CHECK_OPENCL_ERROR(status, "clCreateCommandQueue failed.");


    return SDK_SUCCESS;
}

int MatrixMultiplication::clInitContextFromGL(int argc, char **argv)
{
	cl_int status;

	glutInit(&argc, argv);

	glutInitWindowPosition(100,10);
	glutInitWindowSize(getWidth0(),getHeight1());
	glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
	glutCreateWindow("MatMul");	

	// glutInitContextVersion(4, 3);
	// glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
	// glutInitContextProfile(GLUT_CORE_PROFILE);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
			GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	glutInitWindowSize(m, k);

	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);

	int window_handle = glutCreateWindow("MatMul");

	if (window_handle < 1)
	{
		fprintf(stderr, "ERROR: Could not create a new rendering window.\n");
		exit(EXIT_FAILURE);
	}

	// GL init
	glewInit();
	if (! glewIsSupported("GL_VERSION_2_0 " "GL_ARB_pixel_buffer_object"))
	{
		std::cout << "Support for necessary OpenGL extensions missing."
		<< std::endl;
		return SDK_FAILURE;
	}

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	glViewport(0, 0, m, k);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(
		60.0,
		(GLfloat)m / (GLfloat)k,
		0.1,
		10.0);

    cl_context_properties cpsGL[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
                                      CL_GLX_DISPLAY_KHR, (intptr_t) glXGetCurrentDisplay(),
                                      CL_GL_CONTEXT_KHR, (intptr_t) gGlCtxSep, 0
                                    };
    if (Args->deviceType.compare("gpu") == 0)
    {
        status = clGetGLContextInfoKHR( cpsGL,
                                        CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                                        sizeof(cl_device_id),
                                        &interopDeviceId,
                                        NULL);
        CHECK_OPENCL_ERROR(status, "clGetGLContextInfoKHR failed!!");

        std::cout<<"Interop Device ID is "<<interopDeviceId<<std::endl;

        // Create OpenCL context from device's id
        context = clCreateContext(cpsGL,
                                  1,
                                  &interopDeviceId,
                                  0,
                                  0,
                                  &status);
        CHECK_OPENCL_ERROR(status, "clCreateContext failed.");
    }
    else
    {
        context = clCreateContextFromType(cpsGL,
                                          CL_DEVICE_TYPE_CPU,
                                          NULL,
                                          NULL,
                                          &status);
        CHECK_OPENCL_ERROR(status, "clCreateContextFromType failed!!");
    }

	return SDK_SUCCESS;
}

int MatrixMultiplication::clFreeContext()
{
 	cl_int status;

 	status = clReleaseCommandQueue(commandQueue);
 	CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

 	status = clReleaseContext(context);
 	CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

	return SDK_SUCCESS;	
}

int MatrixMultiplication::glFreeContext()
{
	// Release Context, Command Queue, etc.
	// glXMakeCurrent(displayNameSep, None, NULL);
	// glXDestroyContext(displayNameSep, gGlCtxSep);
	// XDestroyWindow(displayNameSep, winSep);
	// XCloseDisplay(displayNameSep);

	free(devices);

	return SDK_SUCCESS;	
}

int
MatrixMultiplication::glPrepareInput()
{
	// allocate and init memory used by host  input0[width0][height0]
	cl_uint inputSizeBytes0 = width0 * height0 * sizeof(cl_float);

	input0 = (cl_float *) malloc(inputSizeBytes0);
	CHECK_ALLOCATION(input0, "Failed to allocate host memory. (input0)");

	// allocate and init memory used by host input1[width1][height1]
	cl_uint inputSizeBytes1 = width1 * height1 * sizeof(cl_float);

	input1 = (cl_float *) malloc(inputSizeBytes1);
	CHECK_ALLOCATION(input1, "Failed to allocate host memory. (input1)");

	// random initialization of input
	fillRandom<cl_float>(input0, width0, height0, 0, 1);
	fillRandom<cl_float>(input1, width1, height1, 0, 1);

	// Unless quiet mode has been enabled, print the INPUT arrays
	if(!Args->quiet)
	{
		printArray<cl_float>(
			"Input0",
			input0,
			width0,
			1);
		printArray<cl_float>(
			"Input1",
			input1,
			width1,
			1);
	}

	// Return
	return SDK_SUCCESS;
}

int
MatrixMultiplication::glFreeInput()
{
	if (input0)
		free(input0);

	if (input1)
		free(input1);

	return SDK_SUCCESS;
}

int
MatrixMultiplication::glSetupInput()
{
	// If VAO != 0, use existing VAO, otherwise create
	if (vao_id)
		glBindVertexArray(vao_id);
	else
	{
		glGenVertexArrays(1, &vao_id);
		glBindVertexArray(vao_id);     
	}

	// VBOs
	glGenBuffers(1, &vbo_ma);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_ma);
	glBufferData(GL_ARRAY_BUFFER, width0 * height0 * sizeof(cl_float), input0, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

	glGenBuffers(1, &vbo_mb);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_mb);
	glBufferData(GL_ARRAY_BUFFER, width1 * height1 * sizeof(cl_float), input1, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	// Return
	return SDK_SUCCESS;
}

int MatrixMultiplication::glPrepareOutput()
{
	// allocate memory for output[width1][height0]
	cl_uint outputSizeBytes = height0 * width1 * sizeof(cl_float);

	output = (cl_float *)calloc(1, outputSizeBytes);
	CHECK_ALLOCATION(output, "Failed to allocate host memory. (output)");

	// allocate memory for output_pos[2*width1*heigh0] as it contains x and y position
	cl_uint outputposSizeBytes = 2 * outputSizeBytes;

	output_pos = (cl_float *)calloc(1, outputposSizeBytes);
	CHECK_ALLOCATION(output_pos, "Failed to allocate host memory. (output_pos)");

	// FIXME: random initialization position
	// fillRandom<cl_float>(output_pos, 2 * width1, height0, 0, 1);

	// allocate memory for output[width1][height0] of reference implementation
	if(Args->verify)
	{
		verificationOutput = (cl_float *) calloc(1, outputSizeBytes);
		CHECK_ALLOCATION(verificationOutput,
			"Failed to allocate host memory. (verificationOutput)");
	}

	return SDK_SUCCESS;
}

int
MatrixMultiplication::glSetupOutput()
{
	// If VAO != 0, use existing VAO, otherwise create
	if (vao_id)
		glBindVertexArray(vao_id);
	else
	{
		glGenVertexArrays(1, &vao_id);
		glBindVertexArray(vao_id);     
	}

	glGenBuffers(1, &vbo_mc);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_mc);
	glBufferData(GL_ARRAY_BUFFER, width1 * height0 * sizeof(cl_float), output, GL_STATIC_DRAW);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(2);

	glGenBuffers(1, &vbo_pos);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
	glBufferData(GL_ARRAY_BUFFER, width1 * height0 * sizeof(cl_float), output_pos, GL_STATIC_DRAW);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(3);

	return SDK_SUCCESS;
}

int
MatrixMultiplication::glFreeOutput()
{
	if (output)
		free(output);

	if (output_pos)
		free(output_pos);

	return SDK_SUCCESS;
}

int
MatrixMultiplication::glFreeBuffers()
{
	// Releases OpenGL resources
	glDisableVertexAttribArray(3);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDeleteBuffers(1, &vbo_pos);
	glDeleteBuffers(1, &vbo_mc);
	glDeleteBuffers(1, &vbo_mb);
	glDeleteBuffers(1, &vbo_ma);

	glBindVertexArray(0);
	glDeleteVertexArrays(1, &vao_id);

	// Return
	return SDK_SUCCESS;  
}

int
MatrixMultiplication::glSetupProgram()
{
	GLenum error_check_value = glGetError();

	vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader_id, 1, &vertex_shader, NULL);
	glCompileShader(vertex_shader_id);

	fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader_id, 1, &fragment_shader, NULL);
	glCompileShader(fragment_shader_id);

	gl_program_id = glCreateProgram();
	glAttachShader(gl_program_id, vertex_shader_id);
	glAttachShader(gl_program_id, fragment_shader_id);

	glLinkProgram(gl_program_id);
	glUseProgram(gl_program_id);

	error_check_value = glGetError();
	if (error_check_value != GL_NO_ERROR)
	{
		fprintf(stderr, "error: %s: %s\n",
			__FUNCTION__,
			gluErrorString(error_check_value));
		exit(1);
	}

	return SDK_SUCCESS;
}

int
MatrixMultiplication::glFreeProgram()
{
	GLenum error_check_value = glGetError();

	glUseProgram(0);

	glDetachShader(gl_program_id, vertex_shader_id);
	glDetachShader(gl_program_id, fragment_shader_id);

	glDeleteShader(fragment_shader_id);
	glDeleteShader(vertex_shader_id);

	glDeleteProgram(gl_program_id);

	error_check_value = glGetError();
	if (error_check_value != GL_NO_ERROR)
	{
		fprintf(stderr, "error: %s: %s\n",
				__FUNCTION__,
				gluErrorString(error_check_value));
		exit(1);
	}

	return SDK_SUCCESS;
}

int
MatrixMultiplication::clPrepareBuffer()
{
	cl_int status;

	// Create buffer for matrix A from vbo_a
	inputBuffer0 = clCreateFromGLBuffer(
		context,
		CL_MEM_READ_ONLY,
		vbo_ma,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateFromGLBuffer failed. (inputBuffer0)");

	// Create buffer for matrix B from vbo_b
	inputBuffer1 = clCreateFromGLBuffer(
		context,
		CL_MEM_READ_ONLY,
		vbo_mb,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateFromGLBuffer failed. (inputBuffer1)");

	// Create buffer for matrix C from vbo_c
	outputBuffer = clCreateFromGLBuffer(
		context,
		CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		vbo_mc,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateFromGLBuffer failed. (outputBuffer)");

	return SDK_SUCCESS;
}

int
MatrixMultiplication::clFreeBuffer()
{
	cl_int status;

 	status = clReleaseMemObject(inputBuffer0);
 	CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer0)");

 	status = clReleaseMemObject(inputBuffer1);
 	CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer1)");

 	status = clReleaseMemObject(outputBuffer);
 	CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(outputBuffer)");

 	return SDK_SUCCESS;
}

int MatrixMultiplication::clSetupProgram()
{
	cl_int status;

	// create a CL program using the kernel source
	buildProgramData buildData;
	buildData.kernelName = std::string("MatrixMultiplication_Kernels.cl");
	buildData.devices = devices;
	buildData.deviceId = Args->deviceId;
	buildData.flagsStr = std::string("");

	if(Args->isLoadBinaryEnabled())
		buildData.binaryName = std::string(Args->loadBinary.c_str());

	if(Args->isComplierFlagsSpecified())
		buildData.flagsFileName = std::string(Args->flags.c_str());

	int retValue = buildOpenCLProgram(program, context, buildData);
	CHECK_ERROR(retValue, SDK_SUCCESS, "clSetupProgram() failed");

	// If local memory is present then use the specific kernel
	if(lds)
		kernel = clCreateKernel(program, "mmmKernel_local", &status);
	else
		kernel = clCreateKernel(program, "mmmKernel", &status);

	CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");

	return SDK_SUCCESS;    
}

int MatrixMultiplication::clFreeProgram()
{
	// Releases OpenCL resources
 	cl_int status;

 	status = clReleaseKernel(kernel);
 	CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

 	status = clReleaseProgram(program);
 	CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

 	// Return
 	return SDK_SUCCESS;
}

int
MatrixMultiplication::clSetWorkGroupSize()
{
    /*
     * Kernel runs over complete output matrix with blocks of blockSize x blockSize
     * running concurrently
     */
     cl_int status = 0;
     globalThreads[0] = width1 / 4;
     globalThreads[1] = height0/ 4;
     localThreads[0] = blockSize;
     localThreads[1] = blockSize;

	// Setting the KernelWorkGroupInfo values
     status = kernelInfo.setKernelWorkGroupInfo(kernel,
     	devices[Args->deviceId]);
     CHECK_ERROR(status,0, "setKernelWrkGroupInfo failed");

     availableLocalMemory = deviceInfo.localMemSize - kernelInfo.localMemoryUsed;
     neededLocalMemory    = 2 * blockSize * blockSize * sizeof(cl_float);
     if(neededLocalMemory > availableLocalMemory)
     {
     	std::cout << "Unsupported: Insufficient local memory on device." << std::endl;
     	return SDK_SUCCESS;
     }

     if((cl_uint)(localThreads[0] * localThreads[1]) >
     	kernelInfo.kernelWorkGroupSize)
     {
     	if(kernelInfo.kernelWorkGroupSize >= 64)
     	{
     		blockSize = 8;
     		localThreads[0] = blockSize;
     		localThreads[1] = blockSize;
     	}
     	else if(kernelInfo.kernelWorkGroupSize >= 32)
     	{
     		blockSize = 4;
     		localThreads[0] = blockSize;
     		localThreads[1] = blockSize;
     	}
     	else
     	{
     		std::cout << "Out of Resources!" << std::endl;
     		std::cout << "Group Size specified : " << localThreads[0] * localThreads[1] <<
     		std::endl;
     		std::cout << "Max Group Size supported on the kernel : "
     		<< kernelInfo.kernelWorkGroupSize<<std::endl;
     		return SDK_FAILURE;
     	}
     }

     if(localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
     	localThreads[1] > deviceInfo.maxWorkItemSizes[1] ||
     	localThreads[0] * localThreads[1] > deviceInfo.maxWorkGroupSize)
     {
     	std::cout <<
     	"Unsupported: Device does not support requested number of work items." <<
     	std::endl;
     	return SDK_FAILURE;
     }
     return SDK_SUCCESS;
}


int
MatrixMultiplication::clKernelRun()
{
	cl_int   status;
	status = clSetWorkGroupSize();
	CHECK_ERROR(status, SDK_SUCCESS, "clSetWorkGroupSize() failed");

	// Prepare Buffer from GL
	clPrepareBuffer();

	// Acquire GL buffer for inputBuffer 0
	cl_event acquireEvt;
	status = clEnqueueAcquireGLObjects(commandQueue,
		1,
		&inputBuffer0,
		0,
		NULL,
		&acquireEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueAcquireGLObjects failed.");

	status = clFlush(commandQueue);
	CHECK_OPENCL_ERROR(status, "clFlush failed.");

	status = waitForEventAndRelease(&acquireEvt);
	CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(acquireEvt) Failed");

	// Acquire GL buffer for inputBuffer 1
	status = clEnqueueAcquireGLObjects(commandQueue,
		1,
		&inputBuffer1,
		0,
		NULL,
		&acquireEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueAcquireGLObjects failed.");

	status = clFlush(commandQueue);
	CHECK_OPENCL_ERROR(status, "clFlush failed.");

	status = waitForEventAndRelease(&acquireEvt);
	CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(acquireEvt) Failed");

	// Acquire GL buffer for outputBuffer
	status = clEnqueueAcquireGLObjects(commandQueue,
		1,
		&outputBuffer,
		0,
		NULL,
		&acquireEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueAcquireGLObjects failed.");

	status = clFlush(commandQueue);
	CHECK_OPENCL_ERROR(status, "clFlush failed.");

	status = waitForEventAndRelease(&acquireEvt);
	CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(acquireEvt) Failed");

	// Set appropriate arguments to the kernel

	// output array as the 1st argument : stores product of input0 and input1
	status = clSetKernelArg(
		kernel,
		0,
		sizeof(cl_mem),
		(void *)&inputBuffer0);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (inputBuffer0)");

	// the input matrix  as 2nd argument - input0
	status = clSetKernelArg(
		kernel,
		1,
		sizeof(cl_mem),
		(void *)&inputBuffer1);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (inputBuffer1)");

	// the input matrix as 3rd argument - input1
	status = clSetKernelArg(
		kernel,
		2,
		sizeof(cl_mem),
		(void *)&outputBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (outputBuffer)");

	// width0 of the input0 matrix as 4th argument - width0
	status = clSetKernelArg(
		kernel,
		3,
		sizeof(cl_int),
		(void*)&width0);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (width0)");

	// Set local memory argument if Scratchpad is available
	if(lds)
	{
		status = clSetKernelArg(
			kernel,
			4,
			(blockSize * 4) * (blockSize * 4) * sizeof(cl_float),
			NULL);
		CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory)");
	}
	else
	{
		status = clSetKernelArg(kernel, 4, sizeof(cl_int), &width1);
		CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (width1)");
	}

	cl_event ndrEvt;

	// Enqueue a kernel run call
	status = clEnqueueNDRangeKernel(
		commandQueue,
		kernel,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&ndrEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");

	status = clFlush(commandQueue);
	CHECK_OPENCL_ERROR(status, "clFlush failed.");

	// Wait for the kernel call to finish execution
	cl_int eventStatus = CL_QUEUED;
	while(eventStatus != CL_COMPLETE)
	{
		status = clGetEventInfo(
			ndrEvt,
			CL_EVENT_COMMAND_EXECUTION_STATUS,
			sizeof(cl_int),
			&eventStatus,
			NULL);
		CHECK_OPENCL_ERROR(status, "clGetEventInfo failed.");
	}

	status = clReleaseEvent(ndrEvt);
	CHECK_OPENCL_ERROR(status, "clReleaseEvent failed. (ndrEvt)");

	cl_event releaseGLEvt;

	// GL gets control of the memory object inputBuffer0 aka vbo_a
	status = clEnqueueReleaseGLObjects(commandQueue,
		1,
		&inputBuffer0,
		0,
		NULL,
		&releaseGLEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueReleaseGLObjects failed.");

	status = clFlush(commandQueue);
	CHECK_OPENCL_ERROR(status, "clFlush failed.");

	status = waitForEventAndRelease(&releaseGLEvt);
	CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(releaseGLEvt) Failed");

	// GL gets control of the memory object inputBuffer1, aka vbo_b
	status = clEnqueueReleaseGLObjects(commandQueue,
		1,
		&inputBuffer1,
		0,
		NULL,
		&releaseGLEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueReleaseGLObjects failed.");

	status = clFlush(commandQueue);
	CHECK_OPENCL_ERROR(status, "clFlush failed.");

	status = waitForEventAndRelease(&releaseGLEvt);
	CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(releaseGLEvt) Failed");

	// GL gets control of the memory object outputBuffer, aka vbo_c
	status = clEnqueueReleaseGLObjects(commandQueue,
		1,
		&outputBuffer,
		0,
		NULL,
		&releaseGLEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueReleaseGLObjects failed.");

	status = clFlush(commandQueue);
	CHECK_OPENCL_ERROR(status, "clFlush failed.");

	status = waitForEventAndRelease(&releaseGLEvt);
	CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(releaseGLEvt) Failed");

	// Free cl buffers
	clFreeBuffer();

	return SDK_SUCCESS;
}

/*
* This is a naive O(N^3) CPU implementation of matrix multiplication
*/
void
MatrixMultiplication::matrixMultiplicationCPUReference(
	cl_float * output,
	cl_float * input0,
	cl_float * input1,
	const cl_uint y,
	const cl_uint x,
	const cl_uint z)
{
	for(cl_uint i = 0; i < y; i++)
	{
		for(cl_uint j = 0; j < z; j++)
		{
			for(cl_uint k = 0; k < x; k++)
			{
				output[i * z + j] += (input0[i * x + k] * input1[k * z + j]);
			}
		}
	}
}

int
MatrixMultiplication::InitCmdParser()
{
	// Call base class Initialize to get default configuration
	if(Args->initialize() != SDK_SUCCESS)
		return SDK_FAILURE;

	// add an option for getting blockSize from commandline
	Option* xParam = new Option;
	CHECK_ALLOCATION(xParam, "Memory Allocation error.\n");
	xParam->_sVersion = "x";
	xParam->_lVersion = "height0";
	xParam->_description = "height of matrix A";
	xParam->_type     = CA_ARG_INT;
	xParam->_value    = &n;
	Args->AddOption(xParam);
	delete xParam;

	Option* yParam = new Option;
	CHECK_ALLOCATION(yParam, "Memory Allocation error.\n");
	yParam->_sVersion = "y";
	yParam->_lVersion = "width0";
	yParam->_description = "width of matrix A and Height of matrix B";
	yParam->_type     = CA_ARG_INT;
	yParam->_value    = &m;
	Args->AddOption(yParam);
	delete yParam;

	Option* zParam = new Option;
	CHECK_ALLOCATION(zParam, "Memory Allocation error.\n");
	zParam->_sVersion = "z";
	zParam->_lVersion = "width1";
	zParam->_description = "width of matrix B";
	zParam->_type     = CA_ARG_INT;
	zParam->_value    = &k;
	Args->AddOption(zParam);
	delete zParam;

	Option* blockSizeParam = new Option;
	CHECK_ALLOCATION(blockSizeParam, "Memory Allocation error.\n");
	blockSizeParam->_sVersion = "b";
	blockSizeParam->_lVersion = "blockSize";
	blockSizeParam->_description =
	"Use local memory of dimensions blockSize x blockSize";
	blockSizeParam->_type     = CA_ARG_INT;
	blockSizeParam->_value    = &blockSize;
	Args->AddOption(blockSizeParam);
	delete blockSizeParam;

	Option* displayGL = new Option;
	CHECK_ALLOCATION(displayGL, "Memory Allocation error.\n");
	displayGL->_sVersion = "ds";
	displayGL->_lVersion = "displayGL";
	displayGL->_description =
	"Display GL";
	displayGL->_type = CA_NO_ARGUMENT;
	displayGL->_value = &display;
	Args->AddOption(displayGL);
	delete displayGL;
	return SDK_SUCCESS;
}

int
MatrixMultiplication::Setup(int argc, char **argv)
{
	// Validation of input values
	if((n == 0) || (m == 0) || (k == 0))
	{
		std::cout << "Error: Matrix dimensions can not be 0" << std::endl;
		return SDK_FAILURE;
	}

	// Make sure the dimensions are multiples of blockSize
	const int vectorSize = 4;
	if(n % (blockSize * vectorSize) != 0)
	{
		n = (n / (blockSize * vectorSize) + 1) * (blockSize * vectorSize);
	}

	if(m % (blockSize * vectorSize) != 0)
	{
		m = (m / (blockSize * vectorSize) + 1) * (blockSize * vectorSize);
	}

	if(k % (blockSize * vectorSize) != 0)
	{
		k = (k / (blockSize * vectorSize) + 1) * (blockSize * vectorSize);
	}

	width0  = m;
	height0 = n;

	width1  = k;
	height1 = m;

	// Create CL-GL context
	clPrepareContext(argc, argv);

	return SDK_SUCCESS;
}

int
MatrixMultiplication::VerifyResults()
{
	if(Args->verify)
	{
	// reference implementation
		matrixMultiplicationCPUReference(verificationOutput, input0, input1, height0,
			width0,  width1);

	// compare the results and see if they match
		if(compare(output, verificationOutput, height0*width1))
		{
			std::cout<<"Passed!\n" << std::endl;
			return SDK_SUCCESS;
		}
		else
		{
			std::cout<<"Failed\n" << std::endl;
			return SDK_FAILURE;
		}
	}

	return SDK_SUCCESS;
}

int
MatrixMultiplication::Cleanup()
{
	clFreeBuffer();
	clFreeProgram();

	glFreeInput();
	glFreeOutput();
	glFreeBuffers();
	glFreeProgram();
	glFreeContext();

	// Return
	return SDK_SUCCESS;
}


/**
* @brief Initialize GL
*/
void
glInit()
{
	glClearColor(0.0 ,0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	if (me->first_run)
	{
		me->glSetupProgram();

		me->glPrepareInput();
		me->glPrepareOutput();
		me->glSetupInput();
		me->glSetupOutput();

		me->clSetupProgram();
	}
}

/**
* @brief Glut Idle function
*/
void
glIdleFunc()
{
	glutPostRedisplay();
}

/**
* @brief Glut reshape func
*
* @param w width of OpenGL window
* @param h height of OpenGL window
*/
void
glReshapeFunc(int w,int h)
{
	glViewport(0, 0, w, h);
	// glMatrixMode(GL_MODELVIEW);
	// glLoadIdentity();
	// gluPerspective(45.0f, w/h, 1.0f, 100.0f);
	// gluLookAt (0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
}

/**
* @brief OpenGL display function
*/
void 
glDisplayFunc()
{
	glClearColor(0.0 ,0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

	if (me->first_run)
		me->clKernelRun();
	else
	{
		// Need to use new input
		me->glFreeInput();
		me->glPrepareInput();
		me->glSetupInput();

		// Let CL run to get matrix C
		me->clKernelRun();

	}

	// Launch GL Shaders
	glDrawArrays(GL_POINTS, 0, me->getHeight0() * me->getWidth1());

	//Calling kernel for calculating subsequent positions
	glFlush();
	glutSwapBuffers();
}

// keyboard function
void
glKeyboardFunc(unsigned char key, int mouseX, int mouseY)
{
	switch(key)
	{
    	// If the user hits escape or Q, then exit

    	// ESCAPE_KEY = 27
		case 27:
		case 'q':
		case 'Q':
		{
			if(me->Cleanup() != SDK_SUCCESS)
			{
				exit(1);
			}
			else
			{
				exit(0);
			}
		}
		default:
		break;
	}
}

int
main(int argc, char * argv[])
{
	MatrixMultiplication Matmul;
	me = &Matmul;

	if(Matmul.InitCmdParser() != SDK_SUCCESS)
		return SDK_FAILURE;

	if(Matmul.Args->parseCommandLine(argc,
		argv) != SDK_SUCCESS)
		return SDK_FAILURE;

	// Setup 
	if(Matmul.Setup(argc, argv) != SDK_SUCCESS)
		return SDK_FAILURE;

	if(Matmul.getDisplay())
	{
    	// Run in  graphical window if requested
		printf("here\n");
		glInit();
		glutDisplayFunc(glDisplayFunc);
		glutReshapeFunc(glReshapeFunc);
		glutIdleFunc(glIdleFunc);
		glutKeyboardFunc(glKeyboardFunc);
		glutMainLoop();
	}

	// Cleanup
	if(Matmul.Cleanup() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	return SDK_SUCCESS;
}