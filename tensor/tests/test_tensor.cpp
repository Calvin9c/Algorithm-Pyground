#include "tensor/tensor.h"
#include <iostream>
#include <cassert>

void test_cpu_tensor() {
    std::cout << "[TEST] CPU Tensor construction\n";

    Tensor t({2, 3}, DType::from<float>(), Device::CPU());
    assert(t.numel() == 6);
    assert(t.dtype() == DType::from<float>());
    assert(t.device().is_cpu());

    auto* data = t.data_as<float>();
    for (int i = 0; i < t.numel(); ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    std::cout << "✔ CPU Tensor write passed\n";

    t.clear();
    for (int i = 0; i < t.numel(); ++i) {
        assert(data[i] == 0.f);
    }

    std::cout << "✔ CPU Tensor clear passed\n";
}

void test_clone() {
    std::cout << "[TEST] Tensor clone\n";

    Tensor t({2, 2}, DType::from<int>(), Device::CPU());
    auto* data = t.data_as<int>();
    data[0] = 10; data[1] = 20; data[2] = 30; data[3] = 40;

    auto t2 = t.clone();
    auto* data2 = t2.data_as<int>();

    for (int i = 0; i < 4; ++i) {
        assert(data[i] == data2[i]);
    }

    std::cout << "✔ Tensor clone content match\n";
}

void test_to_cuda_and_back() {
    std::cout << "[TEST] CPU ⇄ CUDA transfer\n";

    Tensor cpu_tensor({4}, DType::from<float>(), Device::CPU());
    auto* cpu_data = cpu_tensor.data_as<float>();
    for (int i = 0; i < 4; ++i) cpu_data[i] = static_cast<float>(i + 1);

    Tensor gpu_tensor = cpu_tensor.cuda();  // transfer to CUDA:0
    assert(gpu_tensor.device().is_cuda());

    Tensor back_tensor = gpu_tensor.cpu();  // back to CPU
    assert(back_tensor.device().is_cpu());

    auto* back_data = back_tensor.data_as<float>();
    for (int i = 0; i < 4; ++i) {
        assert(cpu_data[i] == back_data[i]);
    }

    std::cout << "✔ Data match after CPU ⇄ CUDA transfer\n";
}

int main() {
    test_cpu_tensor();
    test_clone();
    test_to_cuda_and_back();

    std::cout << "[PASS] All Tensor tests passed!\n";
    return 0;
}
