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
    NetworkFunction *nf = unet::NetworkFunction::FromOpenCLKernel(
            kernel, "doNothing"
        );

    if (nf != nullptr) {
        delete nf;
    } else {
        return 1;
    }
    return 0;
}
