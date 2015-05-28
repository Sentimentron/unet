#ifndef __H_UNET_OPENCL_DEVICE__
#define __H_UNET_OPENCL_DEVICE__

#ifdef UNET_OPENCL_AVAILABLE
#ifdef APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define STRMAX 1024

namespace unet {
    class OpenCLDevice {
        public:
            cl_device_type deviceType;
            char vendorName[STRMAX];
            char deviceName[STRMAX];
            char deviceVersion[STRMAX];
            char deviceExtensions[STRMAX];
            cl_uint maxComputeUnits;
            cl_ulong globalMemSize;
            size_t maxWorkgroupSize;
            cl_device_local_mem_type localMemType;
            cl_ulong localMemSize;

            static size_t maxGroupWorkSize[3];
        
            // Memory management
            void *AllocateOnDevice(size_t len);
            void CopyToDevice(void *dest, void *src, size_t len);
            void FreeOnDevice(void *mem);

            const char *statusAsString(cl_int status);
            static OpenCLDevice *Initialize(cl_device_id device);
            ~OpenCLDevice();
        private:
            OpenCLDevice();
    }
}

#else
/* Deliberately nothing here */
#endif /*UNET_OPENCL_AVAILABLE */
#endif /*__H_UNET_OPENCL_DEVICE__ */
