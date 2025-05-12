#pragma once

#include "tensor.h"

namespace tensor_kernel {

namespace cpu {
    template<typename Scalar>
    void launch_add_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_mul_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_eq_kernel(const Tensor &a, const Tensor &b, bool &out);
} // namespace cpu

namespace cuda {
    template<typename Scalar>
    void launch_add_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_mul_kernel(const Tensor &a, const Tensor &b, Tensor &out);
    template<typename Scalar>
    void launch_eq_kernel(const Tensor &a, const Tensor &b, bool &out);
} // namespace cuda

}