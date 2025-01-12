#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void convolution(
    uint8_t* input, float* kernel, float* output,
    int img_width, int img_height, int channels,
    int kernel_width, int kernel_height
);

#ifdef __cplusplus
}
#endif


#endif