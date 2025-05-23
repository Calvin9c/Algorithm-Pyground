#pragma once

#include "tensor/tensor.h"

namespace tensor_op {
    Tensor add(const Tensor &a, const Tensor &b);
    Tensor mul(const Tensor &a, const Tensor &b);
    Tensor matmul(const Tensor &a, const Tensor &b);

    bool equal(const Tensor &a, const Tensor &b);
    void cast(const Tensor &src, Tensor &dst);
    
    bool is_contiguous(const Tensor &a);
    Tensor squeeze(const Tensor &src, int64_t d);
    Tensor unsqueeze(const Tensor &src, int64_t d);
    Tensor reshape(const Tensor &src, const std::vector<int64_t> &new_shape);
} // namespace tensor_ops