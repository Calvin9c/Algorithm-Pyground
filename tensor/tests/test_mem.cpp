#include <iostream>
#include <cassert>
#include <cstring>
#include "mem_util.h"
#include "device.h"

using namespace std;

void test_mem(const Device& dst_dev, const Device& src_dev) {
    const size_t nbytes = 16;

    // Allocate and initialize source memory
    void* src = _malloc(nbytes, src_dev);
    cout << "Pass src malloc" << endl;
    _memset(src, 42, nbytes, src_dev);
    cout << "Pass src memset" << endl;

    // Allocate destination memory
    void* dst = _malloc(nbytes, dst_dev);
    cout << "Pass dst malloc" << endl;

    // Perform copy <---
    _memcpy(dst, dst_dev, src, src_dev, nbytes);
    cout << "Pass dst memcpy" << endl;

    // Copy result back to CPU for validation
    uint8_t host_result[nbytes];
    _memcpy(host_result, Device::CPU(), dst, dst_dev, nbytes);

    // Validate content
    for (size_t i = 0; i < nbytes; ++i) {
        assert(host_result[i] == 42);
    }

    std::cout << "[PASS] " << src_dev.str() << " -> " << dst_dev.str() << std::endl;

    _memfree(src, src_dev);
    _memfree(dst, dst_dev);
}

int main() {
    Device cpu = Device::CPU();
    Device cuda0 = Device::CUDA(0);

    test_mem(cpu, cpu);
    test_mem(cuda0, cpu);
    test_mem(cpu, cuda0);
    test_mem(cuda0, cuda0);

    std::cout << "All memory tests passed.\n";
    return 0;
}
