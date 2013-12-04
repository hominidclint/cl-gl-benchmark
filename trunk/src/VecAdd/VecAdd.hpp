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

#ifndef VECADD_H_
#define VECADD_H_

#define GROUP_SIZE 64

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9.214.1"


/**
* VecAdd
* Class implements OpenCL Vector Add

*/

#ifndef COMMON_DECLARE_HPP__
#define COMMON_DECLARE_HPP__
/**
* This file contains the declaration for Vector Add sample.
*/
#ifdef _WIN32
#include <windows.h>
#endif

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

using namespace appsdk;



// GLEW and GLUT includes
#include <GL/glew.h>
#include <CL/cl_gl.h>

#ifdef _WIN32
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")
#pragma warning( disable : 4996)
#endif

// Define DISPLAY_DEVICE_ACTIVE as it is not defined in MinGW
#ifdef _WIN32
#ifndef DISPLAY_DEVICE_ACTIVE
#define DISPLAY_DEVICE_ACTIVE    0x00000001
#endif
#endif

#ifndef _WIN32
#include <GL/glut.h>
#endif

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetGLContextInfoKHR_fn)(
    const cl_context_properties *properties,
    cl_gl_context_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret);

/**
 * Rename references to this dynamically linked function to avoid
 * collision with static link version
 */
#define clGetGLContextInfoKHR clGetGLContextInfoKHR_proc
static clGetGLContextInfoKHR_fn clGetGLContextInfoKHR;

#endif



class VecAdd
{
    public:
        static VecAdd *VecAddGL;

        cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;               /**< time taken to run kernel and read result back */

        cl_context context;                 /**< CL context */
        cl_device_id *devices;              /**< CL device list */
        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program program;                 /**< CL program  */
        cl_kernel kernel;

        cl_uint width;                      /**< Width of window */
        cl_uint height;                     /**< Height of window */

        size_t blockSizeX;                  /**< Work-group size in x-direction */
        size_t blockSizeY;                  /**< Work-group size in y-direction */
        int iterations;                     /**< Number of iterations for kernel execution */
        int factor;
        clock_t t1, t2;
        int frameCount;
        int frameRefCount;
        double totalElapsedTime;

        cl_device_id interopDeviceId;

        SDKDeviceInfo deviceInfo;                /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
        * Constructor
        * Initialize member variables
        */
        VecAdd()
        {
            width = 800;
            height = 600;
            blockSizeX = GROUP_SIZE;
            blockSizeY = 1;
            iterations = 1;
            frameCount = 0;
            frameRefCount = 90;
            totalElapsedTime = 0.0;
            sampleArgs = new CLCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }


        ~VecAdd()
        {
        }

        /**
         * Generate random float number
         */
        float random(float randMax, float randMin);

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
        * OpenCL related initialisations.
        * Set up Context, Device list, Command Queue, Memory buffers
        * Build CL kernel program executable
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupCL();

        /**
        * Initializing GL and get interoperable CL context
        * @param argc number of arguments
        * @param argv command line arguments
        * @
        * @return SDK_SUCCESS on success and SDK_FALIURE on failure.
        */
        int initializeGLAndGetCLContext(cl_platform_id platform,
                                        cl_context &context,
                                        cl_device_id &interopDevice);

        /**
        * Set values for kernels' arguments, enqueue calls to the kernels
        * on to the command queue, wait till end of kernel execution.
        * Get kernel start and end time if timing is enabled
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int runCLKernels();

        /**
        * Override from SDKSample. Print sample stats.
        */
        void printStats();

        /**
        * Override from SDKSample. Initialize
        * command line parser, add custom options
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int initializeCLP();

        /**
        * Override from SDKSample, adjust width and height
        * of execution domain, perform all sample setup
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setup();

        /**
        * Override from SDKSample
        * Run OpenCL Vec Add
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int run();

        /**
        * Override from SDKSample
        * Cleanup memory allocations
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int cleanup();

        /**
        * Override from SDKSample
        * Verify against reference implementation
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int verifyResults();
};
namespace appsdk
{


/**
* buildOpenCLProgram
* builds the opencl program
* @param program program object
* @param context cl_context object
* @param buildData buildProgramData Object
* @return 0 if success else nonzero
*/
int compileOpenCLProgram(cl_program &program, const cl_context& context,
                         buildProgramData &buildData);


}
#endif // VECADD_H_
