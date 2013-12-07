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


#ifndef MATRIXMULTIPLICATION_H_
#define MATRIXMULTIPLICATION_H_


#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glx.h>
#include "CLUtil.hpp"

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9.214.1"

using namespace appsdk;

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

/**
 * MatrixMultiplication
 * Class implements OpenCL Matrix Multiplication sample
 */

 class MatrixMultiplication
 {

		/// Seed
        cl_uint                  seed;                  /**< Seed value for random number generation */

        /// Buffers
        cl_float              *input0;                  /**< Input array */
        cl_int                 width0;                  /**< width of input Array */
        cl_int                height0;                  /**< height of input Array */
        cl_float              *input1;                  /**< Input array */
        cl_int                 width1;                  /**< width of Input Array */
        cl_int                height1;                  /**< height of Input Array */
        cl_float              *output;                  /**< Output Array */
        cl_float  *verificationOutput;                  /**< Output array for reference implementation */
        float      *mappedColorBuffer;

        /// CL context 
        cl_platform_id       platform;
        cl_context            context;                  /**< CL context */
        cl_device_id         *devices;                  /**< CL device list */
        cl_device_id  interopDeviceId;
        cl_command_queue commandQueue;                  /**< CL command queue */
        cl_program            program;                  /**< CL program  */
        cl_kernel              kernel;                  /**< CL kernel */

        /// CL buffers
        cl_mem           inputBuffer0;                  /**< CL memory buffer  for matrix input0*/
        cl_mem           inputBuffer1;                  /**< CL memory buffer  for matrix input1*/
        cl_mem           outputBuffer;                  /**< CL memory buffer  for storing the output*/

        /// CL dimension and others
        cl_uint             blockSize;                  /**< Size of the block used for shared memory */
        cl_int        	            n;                  /**< n height of matrix A and width of matrixB */
        cl_int                      m;                  /**< m width of matrix A */
        cl_int                      k;                  /**< k height of matrix B */
        size_t       globalThreads[2];                  /**< global NDRange */
        size_t        localThreads[2];                  /**< local Threads */

        /// Info
        cl_ulong availableLocalMemory;                  /**< Total Local Memory available for the kernel */
        cl_ulong    neededLocalMemory;                  /**< Total Local Memory needed by the kernel */
        int                iterations;                  /**< Number of iterations for kernel execution */
        SDKDeviceInfo      deviceInfo;                  /**< Structure to store device information*/
        KernelWorkGroupInfo kernelInfo;                 /**< Structure to store kernel related info */

        /// GL VAO and VBOs
        GLuint                 vao_id;                 // This VAO contains below 3 VBOs
        GLuint                 vbo_ma;                 // Containing data for Matrix A
        GLuint                 vbo_mb;                 // Containing data for Matrix B
        GLuint                 vbo_mc;                 // Containing data for Matrix C

        /// GL Shaders
        GLuint       vertex_shader_id;
        GLuint     fragment_shader_id;

        /// Flags
        bool                      lds;                  /**< Local data store availability */
        bool                  display;                  /**< Visualization availability */

        /// CL-GL interoperable context
        int clPrepareContext();
        int clInitContextFromGL(cl_platform_id platform, 
            cl_context &context,
            cl_device_id &interopDevice);

        /// GL
        int glSetupData();
        int glSetupProgram();

        /// CL
        int clSetupData();
        int clSetupProgram();

    public:

        CLCommandArgs   *Args;   /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
         MatrixMultiplication()
         {
            Args = new CLCommandArgs();
            Args->sampleVerStr = SAMPLE_VERSION;
            seed   = 123;
            input0 = NULL;
            input1 = NULL;
            output = NULL;
            verificationOutput = NULL;
            n = 800;
            m = 800;
            k = 600;
            blockSize = 8;
            lds = 0;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
         int setupMatrixMultiplication();

        /**
         * Function to compute local WorkGroup Size based on inputs
         * and device properties
         */
         int setWorkGroupSize();

        /**
         * OpenCL related initialisations.
         * Set up Context, Device list, Command Queue, Memory buffers
         * Build CL kernel program executable
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
         int setupCL();

        /**
         * Set values for kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
         int runCLKernels();

        /**
         * Reference CPU implementation of Matrix Multiplication
         * @param output stores the output of the multiplied matrices depthxheight
         * @param input0 input matrix of size width x height
         * @param input1 input matrix of size depth x width
         * @param height height of the output matrix
         * @param width  length of the common dimension of the matrices input0 and input1
         * @param depth  width  of the output matrix
         */
         void matrixMultiplicationCPUReference(
            cl_float * output,
            cl_float * input0,
            cl_float * input1,
            const cl_uint height,
            const cl_uint width,
            const cl_uint depth);

        // Getters
         cl_float *getOutput() { return output; }
         cl_float *getInput0() { return input0; }
         cl_float *getInput1() { return input1; }

         int getWidth0() { return width0; }
         int getHeight0() { return height0; }
         int getWidth1() { return width1; }
         int getHeight1() { return height1; }
         bool getDisplay() { return display; }


        /**
         * Override from SDKSample. Print sample stats.
         */
         void printStats();

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
         int InitCmdParser();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
         int Setup();

        /**
         * Override from SDKSample
         * Run OpenCL Matrix Multiplication
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



#endif
