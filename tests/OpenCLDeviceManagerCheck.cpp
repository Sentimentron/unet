#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <unet/unet.hpp>
#include <unet/OpenCLDeviceManager.hpp>

using namespace unet;

/// Make sure that an OpenCLDevice can be allocated, 
/// released.
int main (int argc, char **argv) {

    OpenCLDeviceManager::Initialize();
    OpenCLDevice *device = OpenCLDeviceManager::GetDevice();
    device->PrintInfo();
    delete device; // Warning: do not do this

    return 0;
}
