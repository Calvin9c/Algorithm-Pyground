#include "tensor/tensor_op.h"
#include "dtype/dtype.h"
#include "dtype/promote.h"
#include "device/device.h"
#include "tensor/tensor_kernel.h"

namespace {
    void check_inputs(const Tensor& a, const Tensor& b) {
        if (a.sizes() != b.sizes()) {
            throw std::invalid_argument("tensor_op: shape mismatch");
        }
        if (a.scalar_type() == ScalarType::UNKNOWN || b.scalar_type() == ScalarType::UNKNOWN) {
            throw std::runtime_error("tensor_op: dtype is UNKNOWN");
        }
        if (a.device() != b.device()) {
            throw std::invalid_argument("tensor_op: device mismatch");
        }
    }
}

namespace tensor_op {
Tensor add(const Tensor &a, const Tensor &b){

    check_inputs(a, b);
    ScalarType res_scalar_type = 
        promote_scalar_types(a.scalar_type(), b.scalar_type()); 
    
    Tensor res(a.sizes(), DType(res_scalar_type), a.device());
    Tensor a_promoted = a.to(DType(res_scalar_type));
    Tensor b_promoted = b.to(DType(res_scalar_type));  

    if (a_promoted.scalar_type()!=b_promoted.scalar_type()) {
        throw std::runtime_error("add: dtype mismatch");
    }

    if (res.device().is_cpu()){
        DISPATCH_BY_SCALAR_TYPE(res.scalar_type(),
                                tensor_kernel::cpu::launch_add_kernel<cpp_scalar>(a_promoted, b_promoted, res));
    } else if (res.device().is_cuda()) {
        DISPATCH_BY_SCALAR_TYPE(res.scalar_type(),
                                tensor_kernel::cuda::launch_add_kernel<cpp_scalar>(a_promoted, b_promoted, res));
    } else {
        throw std::runtime_error("Invalid device type.");
    }
    return res;
}
Tensor mul(const Tensor &a, const Tensor &b){

    check_inputs(a, b);
    ScalarType res_scalar_type = 
        promote_scalar_types(a.scalar_type(), b.scalar_type());
    
    Tensor res(a.sizes(), DType(res_scalar_type), a.device());
    
    Tensor a_promoted = a.to(DType(res_scalar_type));
    Tensor b_promoted = b.to(DType(res_scalar_type));

    
    if (a_promoted.scalar_type()!=b_promoted.scalar_type()) {
        throw std::runtime_error("mul: dtype mismatch");
    }

    if (res.device().is_cpu()){
        DISPATCH_BY_SCALAR_TYPE(res.scalar_type(),
                                tensor_kernel::cpu::launch_mul_kernel<cpp_scalar>(a_promoted, b_promoted, res));
    } else if (res.device().is_cuda()) {
        DISPATCH_BY_SCALAR_TYPE(res.scalar_type(),
                                tensor_kernel::cuda::launch_mul_kernel<cpp_scalar>(a_promoted, b_promoted, res));
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
        DISPATCH_BY_SCALAR_TYPE(a.scalar_type(),
                                tensor_kernel::cpu::launch_eq_kernel<cpp_scalar>(a, b, res));
    } else if (a.device().is_cuda()) {
        DISPATCH_BY_SCALAR_TYPE(a.scalar_type(),
                                tensor_kernel::cuda::launch_eq_kernel<cpp_scalar>(a, b, res));
    } else {
        throw std::runtime_error("Invalid device type.");
    }
    return res;
}
void cast(const Tensor &src, Tensor &dst) {
    check_inputs(src, dst);
    DISPATCH_BY_SCALAR_TYPE(src.scalar_type(), [&](){
        using SrcScalar = cpp_scalar;
        DISPATCH_BY_SCALAR_TYPE(dst.scalar_type(), [&](){
            using DstScalar = cpp_scalar;
            if (dst.device().is_cpu()) {
                tensor_kernel::cpu::launch_cast_kernel<SrcScalar, DstScalar>(src, dst);
            } else if (dst.device().is_cuda()) {
                tensor_kernel::cuda::launch_cast_kernel<SrcScalar, DstScalar>(src, dst);
            } else {
                throw std::runtime_error("Invalid device type.");
            }                          
        }())
    }())
}
} // namespace tensor_op