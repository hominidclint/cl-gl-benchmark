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


int MatrixMultiplication::clPrepareContext()
{    
    cl_int status = 0;
    cl_device_type dType;

    if(Args->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(Args->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    // Have a look at the available platforms and pick either
    // the AMD one if available or a reasonable default.
    int retValue = getPlatform(platform, Args->platformId,
                               Args->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");

    // creating context
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = clCreateContextFromType(
                  cps,
                  dType,
                  NULL,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR(status, "clCreateContextFromType failed.");

    // getting device on which to run the sample
    status = getDevices(context, &devices, Args->deviceId,
                        Args->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    // Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[Args->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;

        commandQueue = clCreateCommandQueue(
                           context,
                           devices[Args->deviceId],
                           prop,
                           &status);
        CHECK_OPENCL_ERROR(status, "clCreateCommandQueue failed.");
    }

    return SDK_SUCCESS;
}

int MatrixMultiplication::clInitContextFromGL(cl_platform_id platform,
        cl_context &context,
        cl_device_id &interopDevice)
{
    cl_int status = SDK_SUCCESS;
    displayNameSep = XOpenDisplay(NULL);
    int screenNumber = ScreenCount(displayNameSep);
    std::cout<<"Number of displays "<<screenNumber<<std::endl;
    XCloseDisplay(displayNameSep);
    for (int i = 0; i < screenNumber; i++)
    {
        if (Args->isDeviceIdEnabled())
        {
            if (i < (int)Args->deviceId)
            {
                continue;
            }
        }
        char disp[100];
        sprintf(disp, "DISPLAY=:0.%d", i);
        putenv(disp);
        displayNameSep = XOpenDisplay(0);
        int nelements;

        GLXFBConfig *fbc = glXChooseFBConfig(displayNameSep,
                                             DefaultScreen(displayNameSep),
                                             0,
                                             &nelements);

        static int attributeList[] = { GLX_RGBA,
                                       GLX_DOUBLEBUFFER,
                                       GLX_RED_SIZE,
                                       1,
                                       GLX_GREEN_SIZE,
                                       1,
                                       GLX_BLUE_SIZE,
                                       1,
                                       None
                                     };
        XVisualInfo *vi = glXChooseVisual(displayNameSep,
                                          DefaultScreen(displayNameSep),
                                          attributeList);
        XSetWindowAttributes swa;
        swa.colormap = XCreateColormap(displayNameSep,
                                       RootWindow(displayNameSep, vi->screen),
                                       vi->visual,
                                       AllocNone);
        swa.border_pixel = 0;
        swa.event_mask = StructureNotifyMask;
        winSep = XCreateWindow(displayNameSep,
                               RootWindow(displayNameSep, vi->screen),
                               10,
                               10,
                               m,
                               k,
                               0,
                               vi->depth,
                               InputOutput,
                               vi->visual,
                               CWBorderPixel|CWColormap|CWEventMask,
                               &swa);

        XMapWindow (displayNameSep, winSep);
        std::cout << "glXCreateContextAttribsARB "
                  << (void*) glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB")
                  << std::endl;
        GLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB =
            (GLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress((const GLubyte*)
                    "glXCreateContextAttribsARB");

        int attribs[] =
        {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
            GLX_CONTEXT_MINOR_VERSION_ARB, 0,
            0
        };

        GLXContext ctx = glXCreateContextAttribsARB(displayNameSep,
                         *fbc,
                         0,
                         true,
                         attribs);
        glXMakeCurrent (displayNameSep,
                        winSep,
                        ctx);
        gGlCtxSep = glXGetCurrentContext();
        cl_context_properties cpsGL[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
                                          CL_GLX_DISPLAY_KHR, (intptr_t) glXGetCurrentDisplay(),
                                          CL_GL_CONTEXT_KHR, (intptr_t) gGlCtxSep, 0
                                        };
        if (!clGetGLContextInfoKHR)
        {
            clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)
                                    clGetExtensionFunctionAddressForPlatform(platform,"clGetGLContextInfoKHR");
            if (!clGetGLContextInfoKHR)
            {
                std::cout << "Failed to query proc address for clGetGLContextInfoKHR";
            }
        }

        size_t deviceSize = 0;
        status = clGetGLContextInfoKHR(cpsGL,
                                       CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                                       0,
                                       NULL,
                                       &deviceSize);
        CHECK_OPENCL_ERROR(status, "clGetGLContextInfoKHR failed!!");

        int numDevices = (deviceSize / sizeof(cl_device_id));
        std::cout<<"Number of interoperable devices "<<numDevices<<std::endl;
        if(numDevices == 0)
        {
            glXDestroyContext(glXGetCurrentDisplay(), gGlCtxSep);
            continue;
        }
        else
        {
            //Interoperable device found
            std::cout<<"Interoperable device found "<<std::endl;
            break;
        }
    }
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
    // OpenGL animation code goes here
    // GL init
    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 " "GL_ARB_pixel_buffer_object"))
    {
        std::cout << "Support for necessary OpenGL extensions missing."
                  << std::endl;
        return SDK_FAILURE;
    }

    glEnable(GL_TEXTURE_2D);
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
    return SDK_SUCCESS;
}

int
MatrixMultiplication::glSetupData()
{
    // allocate and init memory used by host  input0[width0][height0]
    cl_uint inputSizeBytes0 = width0 * height0 * sizeof(cl_float);

    input0 = (cl_float *) malloc(inputSizeBytes0);
    CHECK_ALLOCATION(input0, "Failed to allocate host memory. (input0)");

    // allocate and init memory used by host input1[width1][height1]
    cl_uint inputSizeBytes1 = width1 * height1 * sizeof(cl_float);

    input1 = (cl_float *) malloc(inputSizeBytes1);
    CHECK_ALLOCATION(input1, "Failed to allocate host memory. (input1)");

    // random initialisation of input
    fillRandom<cl_float>(input0, width0, height0, 0, 1);
    fillRandom<cl_float>(input1, width1, height1, 0, 1);

    // allocate memory for output[width1][height0]
    cl_uint outputSizeBytes = height0 * width1 * sizeof(cl_float);

    output = (cl_float *) malloc(outputSizeBytes);
    CHECK_ALLOCATION(output, "Failed to allocate host memory. (output)");

    // allocate memory for output[width1][height0] of reference implementation
    if(Args->verify)
    {
        verificationOutput = (cl_float *) malloc(outputSizeBytes);
        CHECK_ALLOCATION(verificationOutput,
                         "Failed to allocate host memory. (verificationOutput)");
        memset(verificationOutput, 0, outputSizeBytes);
    }

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

    return SDK_SUCCESS;
}

int
MatrixMultiplication::clSetupData(void)
{

    cl_int status;
    // Set Presistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
    if(Args->isAmdPlatform())
    {
        inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
    }

    // Create buffer for matrix A
    inputBuffer0 = clCreateBuffer(
                       context,
                       inMemFlags,
                       sizeof(cl_float) * width0 * height0,
                       0,
                       &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (inputBuffer0)");

    // Create buffer for matrix B
    inputBuffer1 = clCreateBuffer(
                       context,
                       inMemFlags,
                       sizeof(cl_float) * width1 * height1,
                       0,
                       &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (inputBuffer1)");

    outputBuffer = clCreateBuffer(
                       context,
                       CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                       sizeof(cl_float) * height0 * width1,
                       0,
                       &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (outputBuffer)");

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
    {
        buildData.binaryName = std::string(Args->loadBinary.c_str());
    }

    if(Args->isComplierFlagsSpecified())
    {
        buildData.flagsFileName = std::string(Args->flags.c_str());
    }

    int retValue = buildOpenCLProgram(program, context, buildData);
    CHECK_ERROR(retValue, SDK_SUCCESS, "buildOpenCLProgram() failed");

    // If local memory is present then use the specific kernel
    if(lds)
    {
        kernel = clCreateKernel(program, "mmmKernel_local", &status);
    }
    else
    {
        kernel = clCreateKernel(program, "mmmKernel", &status);
    }
    CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");

    return SDK_SUCCESS;    
}

int
MatrixMultiplication::setWorkGroupSize()
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
MatrixMultiplication::runCLKernels(void)
{
    cl_int   status;
    status = setWorkGroupSize();
    CHECK_ERROR(status, SDK_SUCCESS, "getWorkGroupSize() failed");

    cl_event ndrEvt;
    cl_int eventStatus = CL_QUEUED;


    // Set input data to matrix A and matrix B
    cl_event inMapEvt1, inMapEvt2, inUnmapEvt1, inUnmapEvt2, outMapEvt, outUnmapEvt;
    void* mapPtr1 = clEnqueueMapBuffer(
                        commandQueue,
                        inputBuffer0,
                        CL_FALSE,
                        CL_MAP_WRITE,
                        0,
                        width0 * height0 * sizeof(cl_float),
                        0,
                        NULL,
                        &inMapEvt1,
                        &status);
    CHECK_OPENCL_ERROR(status, "clEnqueueMapBuffer failed. (inputBuffer0)");

    void* mapPtr2 = clEnqueueMapBuffer(
                        commandQueue,
                        inputBuffer1,
                        CL_FALSE,
                        CL_MAP_WRITE,
                        0,
                        width1 * height1 * sizeof(cl_float),
                        0,
                        NULL,
                        &inMapEvt2,
                        &status);
    CHECK_OPENCL_ERROR(status, "clEnqueueMapBuffer failed. (inputBuffer1)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    // random initialisation of input
    fillRandom<cl_float>(input0, width0, height0, 0, 1);
    fillRandom<cl_float>(input1, width1, height1, 0, 1);

    status = waitForEventAndRelease(&inMapEvt1);
    CHECK_ERROR(status,SDK_SUCCESS, "WaitForEventAndRelease(inMapEvt1) Failed");
    memcpy(mapPtr1, input0, sizeof(cl_float) * width0  * height0);

    status = waitForEventAndRelease(&inMapEvt2);
    CHECK_ERROR(status,SDK_SUCCESS, "WaitForEventAndRelease(inMapEvt2) Failed");
    memcpy(mapPtr2, input1, sizeof(cl_float) * width1  * height1);

    status = clEnqueueUnmapMemObject(
                 commandQueue,
                 inputBuffer0,
                 mapPtr1,
                 0,
                 NULL,
                 &inUnmapEvt1);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed. (inputBuffer0)");

    status = clEnqueueUnmapMemObject(
                 commandQueue,
                 inputBuffer1,
                 mapPtr2,
                 0,
                 NULL,
                 &inUnmapEvt2);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed. (inputBuffer1)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&inUnmapEvt1);
    CHECK_ERROR(status, SDK_SUCCESS, "waitForEventAndRelease(inUnmapEvt1) failed");

    status = waitForEventAndRelease(&inUnmapEvt2);
    CHECK_ERROR(status,SDK_SUCCESS, "waitForEventAndRelease(inUnmapEvt2) failed");

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

    // wait for the kernel call to finish execution
    eventStatus = CL_QUEUED;
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

    void* outMapPtr = clEnqueueMapBuffer(
                          commandQueue,
                          outputBuffer,
                          CL_FALSE,
                          CL_MAP_READ,
                          0,
                          width1 * height0 * sizeof(cl_float),
                          0,
                          NULL,
                          &outMapEvt,
                          &status);
    CHECK_OPENCL_ERROR(status, "clEnqueueMapBuffer failed. (outputBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&outMapEvt);
    CHECK_ERROR(status,0, "waitForEventAndRelease(outMapEvt) failed");
    memcpy(output, outMapPtr, sizeof(cl_float) * width1  * height0);

    status = clEnqueueUnmapMemObject(
                 commandQueue,
                 outputBuffer,
                 outMapPtr,
                 0,
                 NULL,
                 &outUnmapEvt);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed. (outputBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&outUnmapEvt);
    CHECK_ERROR(status,0, "waitForEventAndRelease(outUnmapEvt) failed");

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
    {
        return SDK_FAILURE;
    }

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

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory Allocation error.\n");
    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;
    Args->AddOption(num_iterations);
    delete num_iterations;

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
MatrixMultiplication::Setup()
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

    /// Create CL-GL context
    clPrepareContext();
    clInitContextFromGL(platform, context, interopDeviceId);



    return SDK_SUCCESS;
}

int
MatrixMultiplication::run()
{
    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " << iterations << " iterations" <<
              std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        int kernelRun = runCLKernels();
        if(kernelRun != SDK_SUCCESS)
        {
            return kernelRun;
        }
    }

    if(!Args->quiet)
    {
        printArray<cl_float>("Output", output, width1, 1);
    }

    return SDK_SUCCESS;
}

int
MatrixMultiplication::verifyResults()
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

void
MatrixMultiplication::printStats()
{
}

int
MatrixMultiplication::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseMemObject(inputBuffer0);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer0)");

    status = clReleaseMemObject(inputBuffer1);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer1)");

    status = clReleaseMemObject(outputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(outputBuffer)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    // release program resources (input memory etc.)

    FREE(input0);
    FREE(input1);
    FREE(output);
    FREE(verificationOutput);
    FREE(devices);

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
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluPerspective(45.0f, w/h, 1.0f, 100.0f);
    gluLookAt (0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
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

    glPointSize(1.0);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);


    int numPixels = me->getWidth0() * me->getHeight1();
    
    me->runCLKernels();

    if(!me->Args->quiet)
    {
        printArray<cl_float>("Output", me->getOutput(), me->getWidth1(), 1);
    }

    float* r = me->getInput0();
    float* g = me->getInput1();    
    float* b = me->getOutput();

    int scaling = 1000;

    glBegin(GL_POINTS);
    for(int i = 0; i < numPixels; ++i)
    {
        // Set color
        // glColor3f(1.0, 0.0, 0.0);

        glColor3f(r[i], g[i], b[i]);
        //divided by 300 just for scaling
        glVertex4f((i % me->getWidth0()), (i / me->getWidth0()), 0.0, scaling);
        // glVertex4f(output[i] -1.0, output[i] - 1.0, 0.0, 2 * scaling);
    }
    glEnd();

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
        if(me->cleanup() != SDK_SUCCESS)
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
    if(Matmul.Setup() != SDK_SUCCESS)
        return SDK_FAILURE;

    if(Matmul.getDisplay())
    {
        // Run in  graphical window if requested
        glutInit(&argc, argv);
        glutInitWindowPosition(100,10);
        glutInitWindowSize(Matmul.getWidth0(),Matmul.getHeight1());
        glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
        glutCreateWindow("MatMul");
        glInit();
        glutDisplayFunc(glDisplayFunc);
        glutReshapeFunc(glReshapeFunc);
        glutIdleFunc(glIdleFunc);
        glutKeyboardFunc(glKeyboardFunc);
        glutMainLoop();
    }

    // Cleanup
    if(Matmul.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}