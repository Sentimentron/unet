#include <unet/unet.hpp>

using namespace unet;

bool unet::OpenCLSupported() {
#ifdef UNET_OPENCL_AVAILABLE
    return true;
#else
    return false;
#endif
}
