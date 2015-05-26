#ifndef __H_UNET_NETWORK_FUNC__
#define __H_UNET_NETWORK_FUNC__

#include <sys/types.h>
namespace unet {
class NetworkFunction {

    public:
        virtual bool Execute (
                double *in,
                double *out,
                off_t inSize,
                off_t outSize
            ) = 0;


        static NetworkFunction *FromOpenCLKernel(
                const char *kernelSource,
                const char *kernelName
            );

        virtual ~NetworkFunction() = 0;
};
}
#endif
