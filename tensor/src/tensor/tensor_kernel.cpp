#include "tensor/tensor_kernel.h"
#include "tensor/tensor.h"
#include "dtype/dtype.h"
#include <omp.h>
#include <cassert>

namespace host_kernel {

    // ----------
    // add
    // ----------

    template<typename Scalar>
    void add_kernel(const Scalar *a, const Scalar *b, Scalar *out, const int64_t n){
        for (int64_t i = 0; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    // ----------
    // mul
    // ----------

    template<typename Scalar>
    void mul_kernel(const Scalar *a, const Scalar *b, Scalar *out, const int64_t n) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    // ----------
    // eq
    // ----------

    template<typename Scalar>
    void eq_kernel(const Scalar *a, const Scalar *b, bool *out, const int64_t n) {
        bool local_out = true;
        
        #pragma omp parallel for reduction(&: local_out)
        for (int64_t i=0; i<n; ++i) {
            if (a[i]!=b[i]) {
                local_out = false;
            }
        }
        *out = local_out;
    }


    // ----------
    // cast
    // ----------

    template<typename SrcScalar, typename DstScalar>
    void cast_kernel(const SrcScalar* src, DstScalar *dst, const int64_t n) {
        for (int64_t i=0; i<n; ++i) {
            dst[i] = static_cast<DstScalar>(src[i]);
        }
    }

    // ----------
    // dot
    // ----------

    template<typename Scalar>
    void dot_kernel(const Scalar *a, const Scalar *b, Scalar *out, const int64_t K) {
        Scalar s = 0;
        #pragma omp parallel for reduction(+:s)
        for (int64_t i=0; i<K; ++i) {
            s += a[i]*b[i];
        }
        *out = s;
    }

    // ----------
    // mm
    // ----------

    template<typename Scalar>
    void mm_kernel (
        const Scalar *a, const Scalar *b, Scalar *out,
        const int64_t M, const int64_t K, const int64_t N
    ) {
        // Do [M, K] * [K, N]
        #pragma omp parallel for collapse(2)
        for (int64_t i=0; i<M; ++i) {
            for (int64_t j=0; j<N; ++j) {
                Scalar s=0;
                for (int64_t k=0; k<K; ++k) {
                    s += a[i*K+k] * b[k*N+j];
                }
                out[i*N+j] = s;
            }
        }
    }

    template<typename Scalar>
    void mm_kernel_tiled(
        const Scalar *a, const Scalar *b, Scalar *out,
        const int64_t M, const int64_t K, const int64_t N
    ) {
        // Do [M, K] * [K, N]
        constexpr int TILE_M = 64;
        constexpr int TILE_K = 64;
        constexpr int TILE_N = 64;        

        #pragma omp parallel for collapse(2)
        for (int64_t ii = 0; ii < M; ii += TILE_M) {
            for (int64_t jj = 0; jj < N; jj += TILE_N) {
                for (int64_t kk = 0; kk < K; kk += TILE_K) {

                    const int64_t i_max = std::min(ii + TILE_M, M);
                    const int64_t j_max = std::min(jj + TILE_N, N);
                    const int64_t k_max = std::min(kk + TILE_K, K);

                    for (int64_t i = ii; i < i_max; ++i) {
                        for (int64_t j = jj; j < j_max; ++j) {
                            Scalar s = 0;
                            for (int64_t k = kk; k < k_max; ++k) {
                                s += a[i*K+k] * b[k*N+j];
                            }
                            out[i*N+j] += s;
                        }
                    }
                }
            }
        }
    } // end of matmul_kernel_tiled

    // ----------
    // bmm
    // ----------

    template<typename Scalar>
    void bmm_kernel(
        const Scalar *a, const Scalar *b, Scalar *out,
        const int64_t B, const int64_t M, const int64_t K, const int64_t N
    ){
        #pragma omp parallel for
        for (int64_t batch=0; batch<B; ++batch) {
            mm_kernel<Scalar>(
                a+batch*M*K, b+batch*K*N, out+batch*M*N,
                M, K, N 
            );
        }
    }

    template<typename Scalar>
    void bmm_kernel_tiled(
        const Scalar *a, const Scalar *b, Scalar *out,
        const int64_t M, const int64_t K, const int64_t N,
        const int64_t NUM_BATCHS
    ){
        #pragma omp parallel for
        for (int64_t batch=0; batch<NUM_BATCHS; ++batch) {
            mm_kernel_tiled<Scalar>(
                a+batch*M*K, b+batch*K*N, out+batch*M*N,
                M, K, N 
            );
        }
    }

} // namespace anonymous

namespace tensor_kernel /*launcher*/ {
namespace cpu {
    template<typename Scalar>
    void launch_add_kernel(const Tensor &a, const Tensor &b, Tensor &out){
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();
        host_kernel::add_kernel<Scalar>(a_ptr, b_ptr, out_ptr, out.numel());
    }

    template<typename Scalar>
    void launch_mul_kernel(const Tensor &a, const Tensor &b, Tensor &out){
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();
        host_kernel::mul_kernel<Scalar>(a_ptr, b_ptr, out_ptr, out.numel());
    }

    template<typename Scalar>
    void launch_eq_kernel(const Tensor &a, const Tensor &b, bool &out) {
        assert(out==true);
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        bool *out_ptr = &out;
        host_kernel::eq_kernel<Scalar>(a_ptr, b_ptr, out_ptr, a.numel());
    }

    template<typename SrcScalar, typename DstScalar>
    void launch_cast_kernel(const Tensor &src, Tensor &dst) {
        const SrcScalar *src_ptr = src.data_as<SrcScalar>();
        DstScalar *dst_ptr = dst.data_as<DstScalar>();
        host_kernel::cast_kernel<SrcScalar, DstScalar>(src_ptr, dst_ptr, src.numel());
    }

    template<typename Scalar>
    void launch_dot_kernel(const Tensor &a, const Tensor &b, Tensor &out) {
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();
        host_kernel::dot_kernel<Scalar>(a_ptr, b_ptr, out_ptr, a.numel());
    }

    template<typename Scalar>
    void launch_mm_kernel(const Tensor &a, const Tensor &b, Tensor &out) {
        const int64_t M=a.size(-2), K=a.size(-1), N=b.size(-1);
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();
        host_kernel::mm_kernel<Scalar>(a_ptr, b_ptr, out_ptr, M, K, N);
    }

    template<typename Scalar>
    void launch_bmm_kernel(const Tensor &a, const Tensor &b, Tensor &out) {
        const int64_t B=a.size(0), M=a.size(-2), K=a.size(-1), N=b.size(-1);
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();
        host_kernel::bmm_kernel<Scalar>(a_ptr, b_ptr, out_ptr, B, M, K, N);
    }
} // namespace cpu
} // namespace tensor_kernels

// init
#define INSTANTIATION(scalar_type, cpp_scalar, ...) \
template void tensor_kernel::cpu::launch_add_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&);                     \
template void tensor_kernel::cpu::launch_mul_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&);                         \
template void tensor_kernel::cpu::launch_eq_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, bool&);
FOR_EACH_SCALAR_TYPE(INSTANTIATION)
#undef INSTANTIATION

#define INSTANTIATION(scalar_type, cpp_scalar, ...) \
template void tensor_kernel::cpu::launch_dot_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&);                       \
template void tensor_kernel::cpu::launch_mm_kernel<cpp_scalar>(  \
    const Tensor&, const Tensor&, Tensor&);                       \
template void tensor_kernel::cpu::launch_bmm_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&);
MATMUL_SCALAR_TYPE(INSTANTIATION)
#undef INSTANTIATION

#define INSTANTIATION(src_scalar_type, src_scalar, dst_scalar_type, dst_scalar) \
template void tensor_kernel::cpu::launch_cast_kernel<src_scalar, dst_scalar>( \
    const Tensor&, Tensor&);
FOR_EACH_SCALAR_TYPE_PAIR(INSTANTIATION)
#undef INSTANTIATION