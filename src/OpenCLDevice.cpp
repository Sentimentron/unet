#include <string.h>

#ifdef UNET_OPENCL_AVAILABLE
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <unet/OpenCLDevice.hpp>

#include "internal.hpp"

static const char *clGetStatus(cl_int status) {
    switch(status) {
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";

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
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_SUCCESS: return "CL_SUCCESS";
        default:
            return "Unrecognised cl_status";
    }
}

void unet::OpenCLDevice::PrintInfo() {
    Log(INFO, "device vendor: %s", this->vendorName);
    Log(INFO, "device name: %s", this->deviceName);
    Log(INFO, "device version: %s", this->deviceVersion);
    Log(INFO, "device extensions: %s", this->deviceExtensions);
}

cl_kernel unet::OpenCLDevice::CompileKernel(const char *src,
        const char *name) {
   
    cl_int ret; 

    // Find the kernel length
    size_t kernelLength = strlen(src);

    // Create the program 
    cl_program program = clCreateProgramWithSource(
            this->context,
            1,
            (const char **)&src,
            (const size_t *)&kernelLength,
            &ret
    );
    if (ret != CL_SUCCESS) {
        Log(FATAL, "program create error: '%s'", clGetStatus(ret));
        return NULL;
    }

    // Build the kernel program 
    ret = clBuildProgram(
            program,
            1, 
            &this->deviceId,
            NULL,
            NULL,
            NULL
        );
    if (ret != CL_SUCCESS) {
        Log(FATAL, "kernel error '%s'", clGetStatus(ret));
        return NULL;
    }

    cl_kernel kernel = clCreateKernel(program, name, &ret);
    if (ret != CL_SUCCESS) {
        Log(FATAL, "kernel error: '%s'", clGetStatus(ret));
        return NULL;
    }
    
    return kernel;
}

bool unet::OpenCLDevice::AllocateOnDevice(size_t sz, cl_mem *out) {
    cl_int ret;
    *out = clCreateBuffer(
            this->context,      // context
            CL_MEM_READ_WRITE,  // flags
            sz,                 // size
            NULL,               // host pointer
            &ret                // error code
        );
    if (ret != CL_SUCCESS) {
        Log(FATAL, "Device allocation failed (error '%s')", 
                clGetStatus(ret));
        return false;
    }
    return true;
}

bool unet::OpenCLDevice::CopyFromDevice(void *dest, cl_mem src, size_t len) {
    cl_int ret;
    void *buf;
    // Create a cl_mem representing the src pointer
    cl_mem tmp = clCreateBuffer (
            this->context,                           // context
            CL_MEM_WRITE_ONLY,                       // flags
            len,                                     // size
            NULL,                                    // host pointer
            &ret                                     // error code
        );
    if (ret != CL_SUCCESS) {
        Log(FATAL, "clCreateBuffer failed (%s)", clGetStatus(ret));
        return false;
    }

    // Copy the mem somewhere temporary on the device
    ret = clEnqueueCopyBuffer (
            this->commandQueue, // command_queue 
            src,                // src_buffer
            tmp,                // dst_buffer
            0,                  // src_offset
            0,                  // dst_offset
            len,                // cb (byte count)
            0,                  // num_events_in_wait_list
            NULL,               // wait_list
            NULL                // event
        ); 
    bool status = false;
    if (ret != CL_SUCCESS) {
        Log(FATAL, "clEnqueueCopyBuffer failed (%s)", clGetStatus(ret));
        goto release;
    } else {
        // Wait for the copy to complete
        ret = clEnqueueBarrierWithWaitList(
                this->commandQueue, // commandQueue
                0,                  // num_events_in_wait_list
                NULL,               // event_wait_list
                NULL                // event
            );
        UASSERT(ret == CL_SUCCESS);
        status = true;
    }

    // Map the buffer into the host
    buf = clEnqueueMapBuffer (
            this->commandQueue, // command_queue
            tmp,                // buffer
            CL_TRUE,            // blocking_map
            CL_MAP_READ,        // flags
            0,                  // offset
            len,                // len
            0,                  // num_events_in_wait_list
            NULL,               // event_wait_list
            NULL,               // event
            &ret                // errcode_ret
        );
    if (buf == NULL) {
        Log(FATAL, "clEnqueueMapBuffer error, status: %s'", 
                clGetStatus(ret)
        );
        status = false;
        goto release;
    }

    // Copy from temporary device mem to host
    memcpy(dest, buf, len);

    ret = clEnqueueUnmapMemObject (
            this->commandQueue, // command_queue
            tmp,                // memobj
            buf,                // mapped_ptr
            0,                  // num_events_in_wait_list
            NULL,               // event_wait_list
            NULL                // cl_event
        );  
    UASSERT(ret == CL_SUCCESS);


release:
    // Release temporary memory object
    ret = clReleaseMemObject(tmp);
    UASSERT(ret == CL_SUCCESS);
    return status;
}

bool unet::OpenCLDevice::CopyToDevice(cl_mem dest, void *src, size_t len) {
    cl_int ret;
    // Create a cl_mem representing the src pointer
    cl_mem tmp = clCreateBuffer (
            this->context,                          // context
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, // flags
            len,                                    // size
            src,                                    // host pointer
            &ret                                    // error code
        );
    if (ret != CL_SUCCESS) {
        Log(FATAL, "clCreateBuffer failed (%s)", clGetStatus(ret));
        return false;
    }

    // Copy the src mem to the device
    ret = clEnqueueCopyBuffer (
            this->commandQueue, // command_queue 
            tmp,                // src_buffer
            dest,               // dst_buffer
            0,                  // src_offset
            0,                  // dst_offset
            len,                // cb (byte count)
            0,                  // num_events_in_wait_list
            NULL,               // wait_list
            NULL                // event
        ); 
    bool status = false;
    if (ret != CL_SUCCESS) {
        Log(FATAL, "clEnqueueCopyBuffer failed (%s)", clGetStatus(ret));
    } else {
        // Wait for the copy to complete
        // ret = clEnqueueBarrier(this->commandQueue);
        ret = clEnqueueBarrierWithWaitList(
                this->commandQueue, // commandQueue
                0,                  // num_events_in_wait_list
                NULL,               // event_wait_list
                NULL                // event
            );
        UASSERT(ret == CL_SUCCESS);
        status = true;
    } 

    // Release temporary memory object
    ret = clReleaseMemObject(tmp);
    UASSERT(ret == CL_SUCCESS);
    return status;
}

bool unet::OpenCLDevice::FreeOnDevice(cl_mem mem) {
    cl_int ret = clReleaseMemObject(mem);
    return ret == CL_SUCCESS;
}

#define CLINFO(prop, var) do {\
    ret = clGetDeviceInfo( \
            id, prop, sizeof(var), &var, NULL \
        );\
    if (ret != CL_SUCCESS) {\
        Log(FATAL, "Device info error for '%s' (error '%s')", \
                prop, clGetStatus(ret)); \
    } \
} while(0)

unet::OpenCLDevice *unet::OpenCLDevice::Initialize(cl_device_id id) {
    OpenCLDevice *d = new OpenCLDevice();
    if (d == NULL) {
        Log(FATAL, "Allocation failure (OpenCLDevice)");
    } 
    // Phase 1, get some information
    cl_int ret;

    CLINFO(CL_DEVICE_TYPE, d->deviceType);
    CLINFO(CL_DEVICE_VENDOR, d->vendorName);
    CLINFO(CL_DEVICE_NAME, d->deviceName);
    CLINFO(CL_DEVICE_EXTENSIONS, d->deviceExtensions);
    CLINFO(CL_DEVICE_MAX_COMPUTE_UNITS, d->maxComputeUnits);
    CLINFO(CL_DEVICE_GLOBAL_MEM_SIZE, d->globalMemSize);
    CLINFO(CL_DEVICE_LOCAL_MEM_TYPE, d->localMemType);
    CLINFO(CL_DEVICE_LOCAL_MEM_SIZE, d->localMemSize);
    CLINFO(CL_DEVICE_MAX_WORK_GROUP_SIZE, d->maxGroupWorkSize);
    // Phase 2: establish context
    d->context = clCreateContext(
            NULL,   // properties
            1,      // numDevices
            &id,    // devices
            NULL,   // pfn_notify
            NULL,   // user_data
            &ret    // return code
        );
    UASSERT(ret == CL_SUCCESS);
    // Create the command queue
    d->commandQueue = clCreateCommandQueue (
            d->context,    // context 
            id,            // device
            0,             // properties
            &ret          // return code
        );
    UASSERT(ret == CL_SUCCESS);

    d->deviceId = id;
    return d;
}

unet::OpenCLDevice::OpenCLDevice() {
}

unet::OpenCLDevice::~OpenCLDevice() {
    clFlush(this->commandQueue);
    clFinish(this->commandQueue);
    clReleaseCommandQueue(this->commandQueue);
    clReleaseContext(this->context);
}

#else
#endif
