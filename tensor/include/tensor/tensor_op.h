#pragma once

#include "tensor/tensor.h"

namespace tensor_op {
    Tensor add(const Tensor &a, const Tensor &b);
    Tensor mul(const Tensor &a, const Tensor &b);
    bool equal(const Tensor &a, const Tensor &b);
    void cast(const Tensor &src, Tensor &dst);
} // namespace tensor_ops