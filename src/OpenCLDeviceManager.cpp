#include <unet/OpenCLDeviceManager.hpp>
#include "internal.hpp"


#ifdef UNET_OPENCL_AVAILABLE
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

using namespace unet;

static unet::OpenCLDevice *managedDevice;

unet::OpenCLDevice* unet::OpenCLDeviceManager::GetDevice() {
    return managedDevice;
}

void unet::OpenCLDeviceManager::Initialize() {
    // Only support one device for now
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_uint numPlatforms, numDevices;

    // Check for a platform
    cl_int ret = clGetPlatformIDs(1, &platformId, &numPlatforms);
    if (numPlatforms <= 0) {
        Log(FATAL, "No OpenCL platforms exist");
        return;
    }

    // Get the first device
    ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &numDevices);
    if (numDevices <= 0) {
        Log(FATAL, "No OpenCL devices");
        return;
    }

    OpenCLDevice *device = OpenCLDevice::Initialize(deviceId);
    managedDevice = device;
    
}
