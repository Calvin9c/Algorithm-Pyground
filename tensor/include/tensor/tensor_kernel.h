#pragma once

#include "tensor/tensor.h"

namespace tensor_kernel {

namespace cpu {
    template<typename Scalar>
    void launch_add_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_mul_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_eq_kernel(const Tensor &a, const Tensor &b, bool &out);
    template<typename SrcScalar, typename DstScalar>
    void launch_cast_kernel(const Tensor &src, Tensor &dst);
    template<typename Scalar>
    void launch_dot_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_mm_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_bmm_kernel(const Tensor &a, const Tensor &b, Tensor &out);
} // namespace cpu

namespace cuda {
    template<typename Scalar>
    void launch_add_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_mul_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_eq_kernel(const Tensor &a, const Tensor &b, bool &out);
    template<typename SrcScalar, typename DstScalar>
    void launch_cast_kernel(const Tensor &src, Tensor &dst);
    template<typename Scalar>
    void launch_dot_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_mm_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_bmm_kernel(const Tensor &a, const Tensor &b, Tensor &out);
} // namespace cuda

}