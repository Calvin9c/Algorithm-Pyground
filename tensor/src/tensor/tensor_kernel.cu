#include "tensor/tensor.h"
#include "tensor/tensor_kernel.h"
#include "dtype/dtype.h"
#include <cassert>

namespace /*kernel*/ {
    template<typename Scalar>
    __global__ void add_kernel(const Scalar *a, const Scalar *b, Scalar *out, const int64_t n){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            out[i] = a[i] + b[i];
        }
    }

    template<typename Scalar>
    __global__ void mul_kernel(const Scalar *a, const Scalar *b, Scalar *out, const int64_t n){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            out[i] = a[i] * b[i];
        }
    }

    template<typename Scalar>
    __global__ void eq_kernel(const Scalar *a, const Scalar *b, bool *out, const int64_t n){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n && a[i]!=b[i]) {
            *out = false;
        }
    }

    template<typename SrcScalar, typename DstScalar>
    __global__ void cast_kernel(const SrcScalar *src, DstScalar *dst, const int64_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            dst[i] = static_cast<DstScalar>(src[i]);
        }
    }
} // namespace anonymous

namespace tensor_kernel {
namespace cuda {
    template<typename Scalar>
    void launch_add_kernel(const Tensor &a, const Tensor &b, Tensor &out){
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();

        constexpr int block_dim = 256;
        int grid_dim = (out.numel()+block_dim-1) / block_dim;
        add_kernel<Scalar><<<grid_dim, block_dim>>>(a_ptr, b_ptr, out_ptr, out.numel());
        cudaDeviceSynchronize();
    }
    template<typename Scalar>
    void launch_mul_kernel(const Tensor &a, const Tensor &b, Tensor &out){
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();
        
        constexpr int block_dim = 256;
        int grid_dim = (out.numel()+block_dim-1) / block_dim;
        mul_kernel<Scalar><<<grid_dim, block_dim>>>(a_ptr, b_ptr, out_ptr, out.numel());
        cudaDeviceSynchronize();
    }
    template<typename Scalar>
    void launch_eq_kernel(const Tensor &a, const Tensor &b, bool &out){
        assert(out==true);
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        bool *out_ptr = &out;

        constexpr int block_dim = 256;
        int grid_dim = (a.numel()+block_dim-1) / block_dim;
        eq_kernel<Scalar><<<grid_dim, block_dim>>>(a_ptr, b_ptr, out_ptr, a.numel());
        cudaDeviceSynchronize();
    }
    template<typename SrcScalar, typename DstScalar>
    void launch_cast_kernel(const Tensor &src, Tensor &dst){
        const SrcScalar *src_ptr = src.data_as<SrcScalar>();
        DstScalar *dst_ptr = dst.data_as<DstScalar>();
        
        constexpr int block_dim = 256;
        int grid_dim = (src.numel()+block_dim-1) / block_dim;
        cast_kernel<SrcScalar, DstScalar><<<grid_dim, block_dim>>>(src_ptr, dst_ptr, src.numel());
        cudaDeviceSynchronize();
    }
} // namespace cuda
} // namespace tensor_kernel

// init
#define INSTANTIATION(scalar_type, cpp_scalar, ...) \
template void tensor_kernel::cuda::launch_add_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&);                       \
template void tensor_kernel::cuda::launch_mul_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&);                       \
template void tensor_kernel::cuda::launch_eq_kernel<cpp_scalar>(  \
    const Tensor&, const Tensor&, bool&);
FOR_EACH_SCALAR_TYPE(INSTANTIATION)
#undef INSTANTIATION

#define INSTANTIATION(src_scalar_type, src_scalar, dst_scalar_type, dst_scalar) \
template void tensor_kernel::cuda::launch_cast_kernel<src_scalar, dst_scalar>( \
    const Tensor&, Tensor&);
FOR_EACH_SCALAR_TYPE_PAIR(INSTANTIATION)
#undef INSTANTIATION