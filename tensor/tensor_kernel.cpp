#include "tensor_kernel.h"
#include "tensor.h"
#include "dtype.h"
#include <cassert>

namespace /*kernel*/ {
    template<typename Scalar>
    void add_kernel(const Scalar *a, const Scalar *b, Scalar *out, const int64_t n){
        for (int64_t i = 0; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    template<typename Scalar>
    void mul_kernel(const Scalar *a, const Scalar *b, Scalar *out, const int64_t n) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    template<typename Scalar>
    void eq_kernel(const Scalar *a, const Scalar *b, bool *out, const int64_t n) {
        for (int64_t i=0; i<n ;++i) {
            if (a[i] != b[i]) {
                *out = false;
                return;
            }
        }
        *out = true;
    }

} // namespace anonymous

namespace tensor_kernel {
namespace cpu {
    template<typename Scalar>
    void launch_add_kernel(const Tensor &a, const Tensor &b, Tensor &out){
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();
        add_kernel<Scalar>(a_ptr, b_ptr, out_ptr, out.numel());
    }

    template<typename Scalar>
    void launch_mul_kernel(const Tensor &a, const Tensor &b, Tensor &out){
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();
        mul_kernel<Scalar>(a_ptr, b_ptr, out_ptr, out.numel());
    }

    template<typename Scalar>
    void launch_eq_kernel(const Tensor &a, const Tensor &b, bool &out) {
        assert(out==true);
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        bool *out_ptr = &out;
        eq_kernel<Scalar>(a_ptr, b_ptr, out_ptr, a.numel());
    }
} // namespace cpu
} // namespace tensor_kernels

// init
#define OP(scalar_type, cpp_scalar, ...) \
template void tensor_kernel::cpu::launch_add_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&); \
template void tensor_kernel::cpu::launch_mul_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&); \
template void tensor_kernel::cpu::launch_eq_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, bool&);
FOR_EACH_SCALAR_TYPE(OP)
#undef OP