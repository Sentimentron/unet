#ifndef __H_UNET_OPENCL_DEVICE__
#define __H_UNET_OPENCL_DEVICE__

#ifdef UNET_OPENCL_AVAILABLE
#ifdef __APPLE__
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
            cl_device_local_mem_type localMemType;
            cl_ulong localMemSize;

            size_t maxGroupWorkSize[3];
       
            // Device info print 
            void PrintInfo();

            // Memory management
            bool AllocateOnDevice(size_t len, cl_mem *out);
            bool CopyToDevice(cl_mem dest, void *src, size_t len);
            bool CopyFromDevice(void *dest, cl_mem src, size_t len);
            bool FreeOnDevice(cl_mem mem);

            static OpenCLDevice *Initialize(cl_device_id device);
            ~OpenCLDevice();
        private:
            OpenCLDevice();
            cl_context context;
            cl_command_queue commandQueue;
    };
}

#else
/* Deliberately nothing here */
#endif /*UNET_OPENCL_AVAILABLE */
#endif /*__H_UNET_OPENCL_DEVICE__ */
