#include <unet/OpenCLDevice.hpp>
#include <unet/OpenCLDeviceManager.hpp>
#include "internal.hpp"

using namespace unet;

const char *multKernelSource = "__kernel void mmul(\n"
    "const int Mdim, const int Ndim, const int Pdim, \n"
    "__global double* A, __global double *B, __global double *C) {\n"
        "int k; int i = get_global_id(0); int j = get_global_id(1); \n"
        "double tmp = 0.0; for (k = 0; k < pDim; k++) \n"
            "tmp += A[i*Ndim + k] * B[k *Pdim + j];\n"
        "C[i*NDim+j] += tmp;}";

const char *thresholdKernelSource = "__kernel void thresh(\n"
    "const double threshold, __global double *A) {\n"
        "int i = get_global_id(0);\n"
        "if (A[i] > threshold) A[i] = 1.0;\n"
        "else A[i] = 0.0;\n"
    "}";

#define DIM 2

void unet::PropagateXOR(double *input, double *out) {

    cl_int ret;

    // Retrieve just any device we have
    OpenCLDevice *device = OpenCLDeviceManager::GetDevice();

    // 
    // Transfer the weights onto the device
    //
    double firstLayer[] = {1.0, 1.0, 1.0, 1.0};
    double secondLayer[] = {-1.0, 1.0};
    cl_mem weights[2];

    device->AllocateOnDevice(4 * sizeof(double), &weights[0]);
    device->AllocateOnDevice(2 * sizeof(double), &weights[1]);
    device->CopyToDevice(weights[0], firstLayer, sizeof(double) * 4);
    device->CopyToDevice(weights[1], secondLayer, sizeof(double) * 2);

    // No biases are used in this network
    cl_mem layerTransitions[3];
    device->AllocateOnDevice(2 * sizeof(double), &layerTransitions[0]);
    device->AllocateOnDevice(2 * sizeof(double), &layerTransitions[1]);
    device->AllocateOnDevice(sizeof(double), &layerTransitions[2]);

    // Copy the input to the device
    device->CopyToDevice(layerTransitions[0], input, sizeof(double) * 2);

    // Compile a multiplication kernel
    cl_kernel multKernel = device->CompileKernel(multKernelSource, "mmul");

    // Compile a threshold kernel
    // cl_kernel threshKernel = device->CompileKernel(thresholdKernelSource);

    int weight1Dim = 1;
    int weight2Dim = 2;
    // Doing |W| * |I| = |L|, W = |2 x 2| I = |1 x 2|, L = |1 x 2|
    UASSERT(clSetKernelArg(multKernel, 0, sizeof(int), &weight2Dim) == CL_SUCCESS);
    UASSERT(clSetKernelArg(multKernel, 1, sizeof(int), &weight2Dim) == CL_SUCCESS);
    UASSERT(clSetKernelArg(multKernel, 2, sizeof(int), &weight1Dim) == CL_SUCCESS);
    UASSERT(clSetKernelArg(multKernel, 3, sizeof(cl_mem), &weights[0]) == CL_SUCCESS);
    UASSERT(clSetKernelArg(multKernel, 4, sizeof(cl_mem), &layerTransitions[0]) == CL_SUCCESS);
    UASSERT(clSetKernelArg(multKernel, 5, sizeof(cl_mem), &layerTransitions[1]) == CL_SUCCESS);
    
    size_t global[DIM], local[DIM];
    global[0] = weight2Dim; global[1] = (size_t) weight2Dim; 

    ret = clEnqueueNDRangeKernel(device->commandQueue, 
            multKernel,
            weight2Dim,
            NULL,
            global,
            NULL,
            0,
            NULL,
            NULL
        ); 
    clFinish(device->commandQueue);
    UASSERT(ret == CL_SUCCESS);

    // Doing |W| * |I| = |L|, W = |2 x 1| I = |1 x 2|, L = |1 x 1|
    UASSERT(clSetKernelArg(multKernel, 0, sizeof(int), &weight2Dim) == CL_SUCCESS);
    UASSERT(clSetKernelArg(multKernel, 1, sizeof(int), &weight1Dim) == CL_SUCCESS);
    UASSERT(clSetKernelArg(multKernel, 2, sizeof(int), &weight1Dim) == CL_SUCCESS);
    UASSERT(clSetKernelArg(multKernel, 3, sizeof(cl_mem), &weights[1]) == CL_SUCCESS);
    UASSERT(clSetKernelArg(multKernel, 4, sizeof(cl_mem), &layerTransitions[1]) == CL_SUCCESS);
    UASSERT(clSetKernelArg(multKernel, 5, sizeof(cl_mem), &layerTransitions[2]) == CL_SUCCESS);
    
    global[0] = weight2Dim; global[1] = (size_t) weight1Dim; 
    ret = clEnqueueNDRangeKernel(device->commandQueue, 
            multKernel,
            weight2Dim,
            NULL,
            global,
            NULL,
            0,
            NULL,
            NULL
        );
    UASSERT(ret == CL_SUCCESS);
    clFinish(device->commandQueue);
    
    device->CopyFromDevice(out, layerTransitions[2], sizeof(double));
}
