#ifndef __H_UNET_OPENCL_NETWORK_FUNC__
#define __H_UNET_OPENCL_NETWORK_FUNC__
#include <unet/NetworkFunction.hpp>
#include <OpenCL/opencl.h>

namespace unet {
class OpenCLNetworkFunction : NetworkFunction {

    public:
        OpenCLNetworkFunction();
        ~OpenCLNetworkFunction();
        static bool Initialize();
        bool Execute(
                double *in,
                double *out,
                off_t in_size,
                off_t out_size
        ); 

    private:

        cl_program program = NULL;
        cl_kernel kernel = NULL;

        static cl_platform_id platformId;
        static cl_device_id   deviceId;
        static cl_context     context;
        static cl_command_queue commandQueue;
        static cl_uint numDevices;
        static cl_uint numPlatforms;
        static size_t maxWorkGroupSize[3];
        static int activeOpenCLNetworkFunctions;

    friend NetworkFunction;

};
}
#endif
