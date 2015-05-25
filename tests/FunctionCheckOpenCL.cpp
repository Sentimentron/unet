#include <iostream>
#include <unet/unet.hpp>

using namespace unet;

/// OpenCLSupported should check that calling this
/// function works. Run manually on your machine to
/// check the test succeed.
int main(int argc, char **argv) {
    if (OpenCLSupported()) {
        std::cout << "OpenCL is supported on this machine.\n";
    } else {
        std::cout << "OpenCL is not supported on this machine.\n";
    }
    return 0;
}
