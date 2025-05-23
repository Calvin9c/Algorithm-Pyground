#include "tensor/tensor_op.h"
#include "dtype/dtype.h"
#include "dtype/promote.h"
#include "device/device.h"
#include "tensor/tensor_kernel.h"
#include <iostream>
#include <cassert>
#include <unordered_set>

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

    bool can_matmul(const Tensor &a, const Tensor &b) {
        /*
            supppose a with sizes [..., M, K]
            supppose b with sizes [..., K, N]
        */

        if (a.scalar_type() == ScalarType::UNKNOWN || b.scalar_type() == ScalarType::UNKNOWN) {
            return false;
        }

        if (a.device() != b.device()) {
            return false;
        }

        if (a.dim()==0 || b.dim()==0) {
            return false;
        }

        // 1D @ 1D
        if (a.dim()==1 && b.dim()==1) {    
            return a.size(0) == b.size(0);
        }
        
        // 2D @ 1D
        if (a.dim()==2 && b.dim()==1) {
            return a.size(-1) == b.size(0);
        } 
        
        // 1D @ 2D
        if (a.dim()==1 && b.dim()==2) {
            return a.size(0) == b.size(0);
        }

        // ND @ ND
        int64_t a_k = a.size(-1);
        int64_t b_k = b.dim()>=2 ? b.size(-2) : b.size(0);
        if (a_k != b_k) return false;


        // 右對齊 a[:a.dim()-2], b[:b.dim()-2]
        int64_t batch_dims = std::max(a.dim(), b.dim())-2;
        // a_pad, b_pad 必有一個是 0
        int64_t a_pad = batch_dims - (a.dim()-2); 
        int64_t b_pad = batch_dims - (b.dim()-2);

        for (int64_t i = 0; i < batch_dims; ++i) {
            int64_t ad = (i >= a_pad) ? a.size(i-a_pad) : 1;
            int64_t bd = (i >= b_pad) ? b.size(i-b_pad) : 1;
            if (ad != bd && ad != 1 && bd != 1) {
                return false;
            }   
        }
        return true;
    }
    std::vector<int64_t> matmul_res_shape(const Tensor &a, const Tensor &b) {
        // suppose a, b is matmutable
        // 1D @ 1D 
        if (a.dim() == 1 && b.dim() == 1) {
            return {1};
        }

        // 2D @ 1D 
        if (a.dim() == 2 && b.dim() == 1) {
            return {a.size(0)};
        }

        // 1D @ 2D
        if (a.dim() == 1 && b.dim() == 2) {
            return {b.size(-1)};
        }

        // 2D @ 2D 
        if (a.dim() == 2 && b.dim() == 2) {
            return {a.size(-2), b.size(-1)};
        }

        // Batch matmul
        std::vector<int64_t> res;
        int64_t batch_dims = std::max(a.dim(), b.dim())-2;
        int64_t a_pad = batch_dims - (a.dim()-2); 
        int64_t b_pad = batch_dims - (b.dim()-2);

        for (int64_t i=0; i<batch_dims; ++i) {
            int64_t ad = (i >= a_pad) ? a.size(i-a_pad) : 1;
            int64_t bd = (i >= b_pad) ? b.size(i-b_pad) : 1;
            res.push_back(std::max(ad, bd));
        }

        int64_t M = (a.dim() >= 2) ? a.size(a.dim()-2) : 1;
        int64_t N = (b.dim() >= 2) ? b.size(b.dim()-1) : 1;
        res.push_back(M);
        res.push_back(N);

        return res;
    }
    std::vector<int64_t> broadcast_shape(
        const std::vector<int64_t>& a,
        const std::vector<int64_t>& b
    ) {
        std::vector<int64_t> res;

        size_t dim_a=a.size(), dim_b=b.size();
        size_t dim = std::max(dim_a, dim_b);
        int64_t a_pad=dim-dim_a , b_pad=dim-dim_b;
        
        for (size_t i=0; i<dim; ++i) {
            int64_t ad = (i>=a_pad) ? a[i-a_pad] : 1;
            int64_t bd = (i>=b_pad) ? b[i-b_pad] : 1;
            if (ad != bd && ad != 1 && bd != 1)
                throw std::runtime_error("Incompatible broadcast dimension"); 
            res.emplace_back(std::max(ad, bd));
        }
        return res;
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
Tensor matmul(const Tensor &a, const Tensor &b) {

    if (!can_matmul(a, b)) {
        throw std::invalid_argument("tensor_op: can't matmul");
    }
    
    ScalarType res_scalar_type = 
        promote_scalar_types(a.scalar_type(), b.scalar_type());

    if (DType(res_scalar_type)!=algo_pyground::FLOAT32 &&
        DType(res_scalar_type)!=algo_pyground::FLOAT64) {
        throw std::invalid_argument("Unsupport DType.");
    }

    Tensor a_promoted = a.to(DType(res_scalar_type));
    Tensor b_promoted = b.to(DType(res_scalar_type));

    std::vector<int64_t>res_shape = matmul_res_shape(a_promoted, b_promoted);
    Tensor res(res_shape, DType(res_scalar_type), a.device());

    if (a.dim()==b.dim() && b.dim()==1) {
        DISPATCH_TO_KERNEL(res.device_type(), res.scalar_type(), MATMUL_KERNEL,
                           launch_dot_kernel, a, b, res)
    }

    else if (a.dim()==b.dim() && b.dim()==2) {
        DISPATCH_TO_KERNEL(res.device_type(), res.scalar_type(), MATMUL_KERNEL,
                           launch_mm_kernel, a, b, res)
    }

    else if (a.dim()==b.dim() && b.dim()==3) {
        DISPATCH_TO_KERNEL(res.device_type(), res.scalar_type(), MATMUL_KERNEL,
                           launch_bmm_kernel, a, b, res)
    }
    
    else {
        throw std::runtime_error("not support yet");
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
    DISPATCH_BY_SCALAR_TYPE(src.scalar_type(),
        using SrcScalar = cpp_scalar;
        DISPATCH_BY_SCALAR_TYPE(dst.scalar_type(), 
            using DstScalar = cpp_scalar;
            if (dst.device().is_cpu()) {
                tensor_kernel::cpu::launch_cast_kernel<SrcScalar, DstScalar>(src, dst);
            } else if (dst.device().is_cuda()) {
                tensor_kernel::cuda::launch_cast_kernel<SrcScalar, DstScalar>(src, dst);
            } else {
                throw std::runtime_error("Invalid device type.");
            })
    )
}

Tensor squeeze(const Tensor &src, int64_t d) {
    Tensor res(src);
    d = d>0 ? d : d+src.dim();
    if (d<0 || d>src.dim()) {
        throw std::out_of_range("squeeze: dimension out of range");
    }
    if (src.size(d)!=1) {
        throw std::runtime_error("squeeze: dimension size must be 1");
    }
    res.sizes().erase(res.sizes().begin()+d);
    return res;
}
Tensor unsqueeze(const Tensor &src, int64_t d) {
    Tensor res(src);
    d = d>0 ? d : d+src.dim();
    if (d<0 || d>src.dim()) {
        throw std::out_of_range("unsqueeze: dimension out of range");
    }
    if (src.size(d)!=1) {
        throw std::runtime_error("unsqueeze: dimension size must be 1");
    }
    res.sizes().insert(res.sizes().begin()+d, 1);
    return res;
}
Tensor reshape(const Tensor &src, const std::vector<int64_t> &new_shape) {
    Tensor res(src);
    int64_t new_num_elements = 1;
    for (auto &s : new_shape) {
        if (s <= 0) {
            throw std::runtime_error("reshape: dimension sizes must be positive");
        }
        new_num_elements *= s;
    }
    if (src.numel() != new_num_elements) {
        throw std::runtime_error("reshape: numel mismatch");
    }
    res.sizes() = new_shape;
    return res;
}
} // namespace tensor_op