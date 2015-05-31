#include <math.h>
#include <stdio.h>

#include <unet/OpenCLDeviceManager.hpp>
#include <unet/unet.hpp>


int main(int argc, char **argv) {
    
    unet::OpenCLDeviceManager::Initialize();

    double in [4][2] = {{0.0, 0.0},
        {0.0, 1.0}, {1.0, 0.0}, 
        {1.0, 1.0}
    };

    double out[] = {0.0, 1.0, 1.0, 0.0};
    double actual[] = {0.0, 0.0, 0.0, 0.0};

    float ret = 0.0;
    for (int i = 0; i < 4; i++) {
        unet::PropagateXOR(in[i], &actual[i]);
        ret += fabs(actual[i]-out[i]);
        for (int j = 0; j < 4; j++) {
            fprintf(stderr, "%.2f ", in[i][j]);
        }
        fprintf(stderr, "%.2f %.2f\n", out[i], actual[i]);
    }
    return (int)ret;
}
