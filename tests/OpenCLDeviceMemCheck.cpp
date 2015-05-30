#include <stdio.h>
#include <math.h>

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
    
    cl_mem dest;
    const int elemSize = 2048;
    // Allocate memory on device
    if(!device->AllocateOnDevice(elemSize * sizeof(double), &dest)) {
        fprintf(stderr, "bad allocation on device\n");
        return 1;
    }
    // Allocate two arrays on host
    double *srcH = (double *)malloc(elemSize * sizeof(double));
    double *dstH = (double *)malloc(elemSize * sizeof(double));

    // Initialize srcH
    for (int i = 0; i < elemSize; i++) {
        *(srcH + i) = (double)i;
    }

    // Copy to the device
    if (!device->CopyToDevice(dest, srcH, elemSize * sizeof(double))) {
        fprintf(stderr, "bad copy to device\n");
        return 2;
    }

    // Copy back from device
    if (!device->CopyFromDevice(dstH, dest, elemSize * sizeof(double))) {
        fprintf(stderr, "bad copy from device\n");
        return 3;
    }

    // Compare the result
    int returnStatus = 0;
    for (int i = 0; i < elemSize; i++) {
        if (fabs(*(srcH + i) - *(dstH + i)) > 0.01) {
            fprintf(stderr, "bad element at %d %.4f %.4f\n",
                   i, *(srcH + i), *(dstH + i)); 
            returnStatus = 4;
        }
    }

    delete device;

    return returnStatus;

}
