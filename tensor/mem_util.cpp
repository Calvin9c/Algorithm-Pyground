#include "mem_util.h"
#include "device.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>

// --------------------
// CUDA Device Guard
// --------------------
CudaDeviceGuard::CudaDeviceGuard(int target_cuda_dev) {
    CUDA_CHECK(cudaGetDevice(&orig_cuda_dev));
    CUDA_CHECK(cudaSetDevice(target_cuda_dev));
}
CudaDeviceGuard::~CudaDeviceGuard() noexcept(false) {
    CUDA_CHECK(cudaSetDevice(orig_cuda_dev));
}

// --------------------
// Memory interface
// --------------------
void* _malloc(size_t nbytes, const Device& device) {
    if (device.is_cuda()) {
        CudaDeviceGuard guard(device.index());
        return _dev_malloc(nbytes);
    } else /*if (device.is_cpu())*/{
        return _host_malloc(nbytes);
    }
}

void _memfree(void* ptr, const Device& device) {
    if (device.is_cuda()) {
        CudaDeviceGuard guard(device.index());
        _dev_memfree(ptr);
    } else /*if (device.is_cpu())*/ {
        _host_memfree(ptr);
    }
}

void _memset(void* dst, int value, size_t nbytes, const Device& device) {
    if (device.is_cuda()) {
        CudaDeviceGuard guard(device.index());
        _dev_memset(dst, value, nbytes);
    } else /*if (device.is_cpu())*/ {
        _host_memset(dst, value, nbytes);
    }
}

// --------------------
// Host memory
// --------------------
void* _host_malloc(size_t nbytes) {
    void* res = std::malloc(nbytes);
    if (!res && nbytes != 0) {
        throw std::runtime_error("Failed to allocate host memory");
    }
    return res;
}

void _host_memfree(void* ptr) {
    std::free(ptr);
}

void _host_memset(void* dst, int value, size_t nbytes) {
    std::memset(dst, value, nbytes);
}

// --------------------
// Device memory
// --------------------
void* _dev_malloc(size_t nbytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, nbytes));
    return ptr;
}

void _dev_memfree(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void _dev_memset(void* dst, int value, size_t nbytes) {
    CUDA_CHECK(cudaMemset(dst, value, nbytes));
}

// --------------------
// Memory copy
// --------------------
void _memcpy_host2host(void* dst, const void* src, size_t nbytes) {
    std::memcpy(dst, src, nbytes);
}

void _memcpy_host2dev(void* dst, const void* src, size_t nbytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
}

void _memcpy_dev2host(void* dst, const void* src, size_t nbytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
}

void _memcpy_dev2dev(void* dst, int dst_device,
                     const void* src, int src_device,
                     size_t nbytes) {
    if (dst_device == src_device) {
        CudaDeviceGuard guard(dst_device);
        CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
    } else {
        CUDA_CHECK(cudaMemcpyPeer(dst, dst_device, src, src_device, nbytes));
    }
}

void _memcpy(void* dst, const Device& dst_dev,
             const void* src, const Device& src_dev,
             size_t nbytes) {

    if (dst_dev.is_cpu() && src_dev.is_cpu()) {
        _memcpy_host2host(dst, src, nbytes);
    } else if (dst_dev.is_cpu() && src_dev.is_cuda()) {
        CudaDeviceGuard guard(src_dev.index());
        _memcpy_dev2host(dst, src, nbytes);
    } else if (dst_dev.is_cuda() && src_dev.is_cpu()) {
        CudaDeviceGuard guard(dst_dev.index());
        _memcpy_host2dev(dst, src, nbytes);
    } else if (dst_dev.is_cuda() && src_dev.is_cuda()) {
        _memcpy_dev2dev(dst, dst_dev.index(), src, src_dev.index(), nbytes);
    } else {
        throw std::runtime_error("Unsupported device combination in _memcpy");
    }
}
