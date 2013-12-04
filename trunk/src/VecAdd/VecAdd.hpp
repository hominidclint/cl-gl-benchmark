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


#ifndef VECADD_H
#define VECADD_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

#define GROUP_SIZE 128

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9.214.1"

using namespace appsdk;

/**
* VecAdd
* Class implements OpenCL  VecAdd sample
*/

class VecAdd
{
        cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;               /**< time taken to run kernel and read result back */

        cl_context context;                 /**< CL context */
        cl_device_id *devices;              /**< CL device list */

        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program program;                 /**< CL program */
        cl_kernel kernel;                   /**< CL kernel */
        size_t groupSize;                   /**< Work-Group size */

        int iterations;
        SDKDeviceInfo deviceInfo;                /**< Structure to store device information*/
        KernelWorkGroupInfo kernelInfo;          /**< Structure to store kernel related info */

        int fpsTimer;
        int timerNumFrames;

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    private:

        float random(float randMax, float randMin);

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        cl_event glEvent;

        /**
        * Constructor
        * Initialize member variables
        */
        explicit VecAdd()
            : setupTime(0),
              kernelTime(0),
              devices(NULL),
              groupSize(GROUP_SIZE),
              iterations(1),
              fpsTimer(0),
              timerNumFrames(0),
              glEvent(NULL)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        ~VecAdd();

        /**
        * Allocate and initialize host memory array with random values
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupVecAdd();

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
        * Set values for kernels' arguments
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupCLKernels();

        /**
        * Enqueue calls to the kernels
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
        int initialize();

        /**
        * Override from SDKSample, adjust width and height
        * of execution domain, perform all sample setup
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setup();

        /**
        * Override from SDKSample
        * Run OpenCL VecAdd
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


        // init the timer for FPS calculation
        void initFPSTimer()
        {
            timerNumFrames = 0;
            fpsTimer = sampleTimer->createTimer();
            sampleTimer->resetTimer(fpsTimer);
            sampleTimer->startTimer(fpsTimer);
        };

        // calculate FPS
        double getFPS()
        {
            sampleTimer->stopTimer(fpsTimer);
            double elapsedTime = sampleTimer->readTimer(fpsTimer);
            double fps = timerNumFrames/elapsedTime;
            timerNumFrames = 0;
            sampleTimer->resetTimer(fpsTimer);
            sampleTimer->startTimer(fpsTimer);
            return fps;
        };
};

#endif // VECADD_H
