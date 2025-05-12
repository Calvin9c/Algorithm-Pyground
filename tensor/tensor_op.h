#pragma once

#include "tensor.h"

namespace tensor_op {
    Tensor add(const Tensor &a, const Tensor &b);
    Tensor mul(const Tensor &a, const Tensor &b);
    bool equal(const Tensor &a, const Tensor &b);
} // namespace tensor_ops