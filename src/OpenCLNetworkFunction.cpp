#include "OpenCLNetworkFunction.hpp"
#include "internal.hpp"
#include <string.h>

using namespace unet; 

cl_platform_id OpenCLNetworkFunction::platformId = NULL;
cl_device_id OpenCLNetworkFunction::deviceId = NULL;
cl_context OpenCLNetworkFunction::context = NULL;
cl_command_queue OpenCLNetworkFunction::commandQueue = NULL;
int OpenCLNetworkFunction::activeOpenCLNetworkFunctions = 0;
cl_uint OpenCLNetworkFunction::numDevices = 0;
cl_uint OpenCLNetworkFunction::numPlatforms = 0;

static const char *clGetStatus(cl_int status) {
    switch(status) {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        default:
            return "Unrecognised cl_status";
    }
}

bool unet::OpenCLNetworkFunction::Initialize() {
    cl_int ret; 

    if (activeOpenCLNetworkFunctions > 0) return true;

    // Retrieve OpenCL device information
    ret = clGetPlatformIDs(1, &platformId, &numPlatforms);
    UASSERT(ret == CL_SUCCESS);
    ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &numDevices);
    UASSERT(ret == CL_SUCCESS);
    // Create OpenCL context
    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &ret);
    UASSERT(ret == CL_SUCCESS);
    // Create command queue
    commandQueue = clCreateCommandQueue(context, deviceId, 0, &ret);
    UASSERT(ret == CL_SUCCESS);

    // Housekeeping
    activeOpenCLNetworkFunctions = 1; 

    return true;
}

OpenCLNetworkFunction::OpenCLNetworkFunction() {
    activeOpenCLNetworkFunctions++;
}

OpenCLNetworkFunction::~OpenCLNetworkFunction() {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    activeOpenCLNetworkFunctions--;
    if (OpenCLNetworkFunction::activeOpenCLNetworkFunctions <= 0) {
        clFlush(commandQueue);
        clFinish(commandQueue);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }
}

bool OpenCLNetworkFunction::Execute (
        double *in, double *out, off_t in_size, off_t out_size
    ) {
    cl_int ret;
    cl_event event;
    UASSERT(in_size == out_size);
    // Assumption: in and out are both on the device 
    cl_mem inputMem = gcl_create_buffer_from_ptr(in);
    cl_mem outputMem = gcl_create_buffer_from_ptr(out);
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputMem);
    UASSERT(ret == CL_SUCCESS);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputMem);
    UASSERT(ret == CL_SUCCESS);
    
    size_t globalWorkSize[3] = {(size_t)in_size, 0, 0};
    size_t localWorkSize[3] = {(size_t)in_size, 0, 0};

    ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    UASSERT(ret == CL_SUCCESS);

    clWaitForEvents(1, &event);
    return true;
}

NetworkFunction* unet::NetworkFunction::FromOpenCLKernel (
        const char *kernelSource,
        const char *kernelName
) {
    cl_int ret;
    OpenCLNetworkFunction *out = NULL;
    OpenCLNetworkFunction::Initialize();

    // Create return structure
    out = new OpenCLNetworkFunction(); 
    
    // Estimate kernel length 
    size_t kernelLength = strlen(kernelSource);

    // Create the program 
    cl_program program = clCreateProgramWithSource(
            OpenCLNetworkFunction::context, 
            1, 
            (const char **)&kernelSource, 
            (const size_t *)&kernelLength, 
            &ret
        );
    UASSERT(ret == CL_SUCCESS);

    // Build kernel program 
    ret = clBuildProgram(
            program, 
            1, 
            &OpenCLNetworkFunction::deviceId, 
            NULL, 
            NULL, 
            NULL
        );
    UASSERT(ret == CL_SUCCESS);

    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, kernelName, &ret);
    if (ret != CL_SUCCESS) {
        Log(FATAL, "clCreateKernel failure: error was '%s'", 
                clGetStatus(ret));
        return nullptr;
    }
    out->program = program;
    out->kernel = kernel;

    return out;
}
