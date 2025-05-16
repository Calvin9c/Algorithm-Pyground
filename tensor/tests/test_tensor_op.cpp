#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include <iostream>
#include <cassert>

void fill_tensor(Tensor& t, float offset = 0.f) {
    auto* data = t.data_as<float>();
    for (int i = 0; i < t.numel(); ++i) {
        data[i] = static_cast<float>(i) + offset;
    }
}

void assert_equal(const Tensor& a, const Tensor& b, float eps = 1e-5f) {
    assert(a.dtype() == b.dtype());
    assert(a.sizes() == b.sizes());

    const float* a_ptr = a.data_as<float>();
    const float* b_ptr = b.data_as<float>();

    for (int i = 0; i < a.numel(); ++i) {
        float diff = std::abs(a_ptr[i] - b_ptr[i]);
        assert(diff < eps);
    }
}

void test_add_mul_cpu_cuda_consistency() {
    std::cout << "[TEST] Tensor add/mul across CPU and CUDA\n";
    
    Tensor a_cpu({4}, DType::from<float>(), Device::CPU());
    Tensor b_cpu({4}, DType::from<float>(), Device::CPU());

    fill_tensor(a_cpu, 0.f);   // [0,1,2,3]
    fill_tensor(b_cpu, 10.f);  // [10,11,12,13]

    Tensor add_cpu = tensor_op::add(a_cpu, b_cpu); // [10,12,14,16]

    Tensor mul_cpu = tensor_op::mul(a_cpu, b_cpu); // [0,11,24,39]

    Tensor a_cuda = a_cpu.cuda();
    Tensor b_cuda = b_cpu.cuda();

    Tensor add_cuda = tensor_op::add(a_cuda, b_cuda).cpu(); // back to CPU
    Tensor mul_cuda = tensor_op::mul(a_cuda, b_cuda).cpu();

    assert_equal(add_cpu, add_cuda);
    assert_equal(mul_cpu, mul_cuda);

    std::cout << "✔ CPU and CUDA add/mul results match\n";
}

int main() {
    test_add_mul_cpu_cuda_consistency();
    std::cout << "[PASS] All tensor_op tests passed!\n";
    return 0;
}
