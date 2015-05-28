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
size_t OpenCLNetworkFunction::maxWorkGroupSize[3] = {0, 0, 0};

static const char *clGetStatus(cl_int status) {
    switch(status) {
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_SUCCESS: return "CL_SUCCESS";
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

    // Retrieve the maximum work-group size
    ret = clGetDeviceInfo(
        deviceId,
        CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(maxWorkGroupSize),
        &maxWorkGroupSize,
        NULL
    );
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

    if (in_size >= OpenCLNetworkFunction::maxWorkGroupSize[0]) {
        Log(FATAL, "Maximum size is %d", OpenCLNetworkFunction::maxWorkGroupSize[0]);
    }

    // Assumption: in and out are both on the device
    cl_mem inputMem = gcl_create_buffer_from_ptr(in);
    cl_mem outputMem = gcl_create_buffer_from_ptr(out);
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), in);
    if (ret != CL_SUCCESS) {
        Log(FATAL, "clSetKernelArg error: %s", clGetStatus(ret));
        return false;
    }

    size_t globalWorkSize[3] = {(size_t)in_size, 0, 0};
    size_t localWorkSize[3] = {(size_t)in_size, 0, 0};

    ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        Log(FATAL, "clEnqueNDRangeKernel error: %s", clGetStatus(ret));
        return false;
    }
    UASSERT(ret == CL_SUCCESS);
    clFinish(commandQueue);
    //clWaitForEvents(1, &event);
    clReleaseMemObject(inputMem);
    clReleaseMemObject(outputMem);
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
