#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include "device.h"

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                     cudaGetErrorString(err));     \
        }                                                         \
    } while (0)

// --------------------
// CUDA Device Guard
// --------------------
class CudaDeviceGuard {
    public:
        explicit CudaDeviceGuard(int target_cuda_dev);
        ~CudaDeviceGuard() noexcept(false);
    private:
        int orig_cuda_dev;
};

// --------------------
// Memory interface
// --------------------
void* _malloc(size_t nbytes, const Device& device);
void _memfree(void* ptr, const Device& device);
void _memset(void* dst, int value, size_t nbytes, const Device& device);

// --------------------
// Host memory
// --------------------
void* _host_malloc(size_t nbytes);
void _host_memfree(void* ptr);
void _host_memset(void* dst, int value, size_t nbytes);

// --------------------
// Device memory
// --------------------
void* _dev_malloc(size_t nbytes);
void _dev_memfree(void* ptr);
void _dev_memset(void* dst, int value, size_t nbytes);

// --------------------
// Memory copy
// --------------------
void _memcpy(void* dst, const Device& dst_dev,
             const void* src, const Device& src_dev,
             size_t nbytes);
void _memcpy_host2host(void* dst, const void* src, size_t nbytes);
void _memcpy_host2dev(void* dst, const void* src, size_t nbytes);
void _memcpy_dev2host(void* dst, const void* src, size_t nbytes);
void _memcpy_dev2dev(void* dst, int dst_device,
                     const void* src, int src_device,
                     size_t nbytes);