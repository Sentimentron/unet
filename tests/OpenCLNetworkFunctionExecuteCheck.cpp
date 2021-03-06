#include <math.h>
#include <OpenCL/opencl.h>

#include <iostream>

#include <unet/unet.hpp>
#include <unet/NetworkFunction.hpp>

using namespace unet;

const char *kernel = "__kernel void doNothing(__global float* a) {\n"
    "int gid = get_global_id(0);\n"
    "a[gid] += a[gid];\n"
    "}";

/// Check that we can create and release OpenCLNetworkFunction
/// objects without leaking resources.
int main(int argc, char **argv) {
    dispatch_queue_t queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    NetworkFunction *nf = unet::NetworkFunction::FromOpenCLKernel(
            kernel, "doNothing"
        );

    // Create arrays
    const int elements = 64; 
    double *in = (double *)malloc(
            elements * sizeof(double)
        );
    for (int i = 0; i < elements; i++) {
        *(in + i) = (double)i;
    }

    double *out = (double *)malloc(
            elements * sizeof(double)
        );

    double *din = (double *)gcl_malloc(
            elements * sizeof(double),
            in,
            CL_MEM_COPY_HOST_PTR
        );
    if (din == NULL) {
        fprintf(stderr, "gcl_malloc error!\n");
        return 1;
    }

    // Run the kernel
    if (!nf->Execute(din, din, elements, elements)) {
        fprintf(stderr, "Execution failure!\n");
        return 1;
    }
    
    dispatch_sync(queue, ^{
        gcl_memcpy(out, din, sizeof(double) * elements);
    });

    // Check the result, should be exactly the same
    for (int i = 0; i < elements; i++) {
        if (fabs(*(in + i) - *(out + i)) > 0.001) {
            fprintf(stderr, "Element %d error (%.5f %.5f)\n",
                    i, *(in + i), *(out + i)
                );
            return 1;
        }
    }

    free(in);
    free(out);
    gcl_free(din);

    if (nf != nullptr) {
        delete nf;
    } else {
        return 1;
    }
    return 0;
}
