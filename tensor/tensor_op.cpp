#include "tensor_op.h"
#include "dtype.h"
#include "device.h"
#include "tensor_kernel.h"

namespace {
    void check_inputs(const Tensor& a, const Tensor& b) {
        if (a.sizes() != b.sizes()) {
            throw std::invalid_argument("tensor_op: shape mismatch");
        }
        if (a.dtype() != b.dtype()) {
            throw std::invalid_argument("tensor_op: dtype mismatch");
        }
        if (a.dtype().type() == ScalarType::UNKNOWN) {
            throw std::runtime_error("Tensor::op: dtype is UNKNOWN");
        }
        if (a.device() != b.device()) {
            throw std::invalid_argument("tensor_op: device mismatch");
        }
    }
}

namespace tensor_op {
Tensor add(const Tensor &a, const Tensor &b){

    check_inputs(a, b);

    Tensor res(a.sizes(), a.dtype(), a.device());
    
    if (res.device().is_cpu()){
        DISPATCH_BY_SCALAR_TYPE(res.dtype().type(),
                                tensor_kernel::cpu::launch_add_kernel<cpp_scalar>(a, b, res));
    } else if (res.device().is_cuda()) {
        DISPATCH_BY_SCALAR_TYPE(res.dtype().type(),
                                tensor_kernel::cuda::launch_add_kernel<cpp_scalar>(a, b, res));
    } else {
        throw std::runtime_error("Invalid device type.");
    }
    return res;
}
Tensor mul(const Tensor &a, const Tensor &b){

    check_inputs(a, b);

    Tensor res(a.sizes(), a.dtype(), a.device());

    if (res.device().is_cpu()){
        DISPATCH_BY_SCALAR_TYPE(res.dtype().type(),
                                tensor_kernel::cpu::launch_mul_kernel<cpp_scalar>(a, b, res));
    } else if (res.device().is_cuda()) {
        DISPATCH_BY_SCALAR_TYPE(res.dtype().type(),
                                tensor_kernel::cuda::launch_mul_kernel<cpp_scalar>(a, b, res));
    } else {
        throw std::runtime_error("Invalid device type.");
    }

    return res;
}
bool equal(const Tensor &a, const Tensor &b){

    if (a.sizes() != b.sizes()) {
        return false;
    } else if (a.dtype() != b.dtype()) {
        return false;
    } else if (a.device() != b.device()) {
        return false;
    }

    bool res = true;
    if (a.device().is_cpu()) {
        DISPATCH_BY_SCALAR_TYPE(a.dtype().type(),
                                tensor_kernel::cpu::launch_eq_kernel<cpp_scalar>(a, b, res));
    } else if (a.device().is_cuda()) {
        DISPATCH_BY_SCALAR_TYPE(a.dtype().type(),
                                tensor_kernel::cuda::launch_eq_kernel<cpp_scalar>(a, b, res));
    } else {
        throw std::runtime_error("Invalid device type.");
    }
    return res;
}
} // namespace tensor_op