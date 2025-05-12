#include <iostream>
#include <cassert>
#include <vector>
#include "storage.h"
#include "device.h"

void test_cpu_storage() {
    std::cout << "[TEST] Storage (CPU) basic operations\n";

    size_t bytes = 16;
    Storage storage(bytes, Device::CPU());

    assert(storage.data() != nullptr);
    assert(storage.nbytes() == bytes);
    assert(storage.device().is_cpu());
    std::cout << "  ✔ constructor passed\n";

    storage.clear();
    auto* data = storage.data_as<uint8_t>();
    for (size_t i = 0; i < bytes; ++i) {
        assert(data[i] == 0);
    }
    std::cout << "  ✔ clear passed\n";

    for (size_t i = 0; i < bytes; ++i) {
        data[i] = static_cast<uint8_t>(i + 1);
    }

    auto clone = storage.clone();
    auto* clone_data = clone->data_as<uint8_t>();
    for (size_t i = 0; i < bytes; ++i) {
        assert(clone_data[i] == data[i]);
    }
    std::cout << "  ✔ clone passed\n";

    size_t new_bytes = 32;
    storage.resize(new_bytes);
    auto* resized_data = storage.data_as<uint8_t>();
    for (size_t i = 0; i < bytes; ++i) {
        assert(resized_data[i] == i + 1);
    }
    std::cout << "  ✔ resize passed\n";
}

void test_gpu_storage() {
    std::cout << "[TEST] Storage (CUDA) basic operations\n";

    size_t bytes = 16;
    Storage gpu_storage(bytes, Device::CUDA());

    assert(gpu_storage.data() != nullptr);
    assert(gpu_storage.nbytes() == bytes);
    assert(gpu_storage.device().is_cuda());
    std::cout << "  ✔ constructor passed\n";

    gpu_storage.clear();

    // host side reference
    std::vector<uint8_t> host_data(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        host_data[i] = static_cast<uint8_t>(i + 1);
    }

    // 拷貝資料上 GPU
    _memcpy(gpu_storage.data(), gpu_storage.device(), host_data.data(), Device::CPU(), bytes);

    // clone 測試
    auto clone = gpu_storage.clone();
    std::vector<uint8_t> clone_data(bytes);
    _memcpy(clone_data.data(), Device::CPU(), clone->data(), clone->device(), bytes);
    for (size_t i = 0; i < bytes; ++i) {
        assert(clone_data[i] == host_data[i]);
    }
    std::cout << "  ✔ clone passed\n";

    // resize 測試
    gpu_storage.resize(32);
    assert(gpu_storage.nbytes() == 32);

    std::vector<uint8_t> resized_data(bytes);
    _memcpy(resized_data.data(), Device::CPU(), gpu_storage.data(), gpu_storage.device(), bytes);
    for (size_t i = 0; i < bytes; ++i) {
        assert(resized_data[i] == host_data[i]);
    }
    std::cout << "  ✔ resize passed\n";
}

int main() {
    test_cpu_storage();
    test_gpu_storage();
    std::cout << "[PASS] All Storage tests (CPU + GPU) passed!\n";
    return 0;
}
