#include "convolution.h"
#include <cuda_runtime.h>

__global__ void convolution_kernel(
    uint8_t* input, float* kernel, float* output,
    int img_width, int img_height, int channels,
    int kernel_width, int kernel_height
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_width && y < img_height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            int k_half_w = kernel_width / 2, k_half_h = kernel_height / 2;
            for (int ky = -k_half_h; ky <= k_half_h; ky++) {
                for (int kx = -k_half_w; kx <= k_half_w; kx++) {
                    // position of image pixel
                    int ix = min(max(x + kx, 0), img_width - 1);
                    int iy = min(max(y + ky, 0), img_height - 1);
                    sum += input[(iy * img_width + ix) * channels + c] * kernel[(ky + k_half_h) * kernel_width + (kx + k_half_w)];
                }
            }
            output[(y * img_width + x) * channels + c] = sum;
        }
    }
}

__host__
void convolution(
    uint8_t* input, float* kernel, float* output,
    int img_width, int img_height, int channels,
    int kernel_width, int kernel_height
){
    // allocate GPU memory
    uint8_t *d_input;
    float *d_kernel, *d_output;

    size_t img_size = img_width * img_height * channels * sizeof(uint8_t);
    size_t kernel_size = kernel_width * kernel_height * sizeof(float);
    size_t output_size = img_width * img_height * channels * sizeof(float);

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_output, output_size);

    // copy data to GPU
    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);

    // execute Kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((img_width + 15) / 16, (img_height + 15) / 16);
    convolution_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, img_width, img_height, channels, kernel_width, kernel_height);

    // copy the result from GPU to CPU
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
