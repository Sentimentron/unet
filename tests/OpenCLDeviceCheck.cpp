#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <unet/unet.hpp>
#include <unet/OpenCLDevice.hpp>

using namespace unet;

/// Make sure that an OpenCLDevice can be allocated, 
/// released.
int main (int argc, char **argv) {

    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_uint numPlatforms, numDevices;

    // Retrieve OpenCL device inforamation
    cl_int ret = clGetPlatformIDs(1, &platformId, &numPlatforms);
    if (numPlatforms <= 0) {
        fprintf(stderr, "not enough platforms!\n");
        return 1;
    }
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Bad status code\n");
        return ret;
    }

    ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &numDevices);
    if (numDevices <= 0) {
        fprintf(stderr, "not enough devices!");
    }
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Bad status code\n");
        return ret;
    }

    OpenCLDevice *device = OpenCLDevice::Initialize(deviceId);
    device->PrintInfo();
    delete device;

    return 0;

}
