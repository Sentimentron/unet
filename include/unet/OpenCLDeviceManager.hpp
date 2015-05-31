#ifndef __H_UNET_OPEN_DEVICE__
#define __H_UNET_OPEN_DEVICE__

#include <unet/OpenCLDevice.hpp>
#include <vector>

namespace unet {
    
    class OpenCLDeviceManager {
    
        public:
            static void Initialize();
            static OpenCLDevice *GetDevice();

    };

}


#endif
