# Tensor
This directory contains a lightweight implementation of a `Tensor` class, loosely modeled after PyTorch's design. It supports basic CPU/GPU memory management and simple arithmetic operations such as addition and multiplication.

Supported data types include: `int8_t`, `int16_t`, `int32_t`, `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`, `float`, and `double`. For more details, see [`include/dtype/dtype.h`](include/dtype/dtype.h).

This project serves as a personal exercise to better understand how tensor systems can be built from the ground up in C++ and CUDA. It is not intended to replicate the full features or performance of mature libraries.

## Installation
### 1. Build Static Library (`libtensor.a`)
```bash
# Navigate to Algorithm-Pyground/tensor
cmake -B build -S .
cmake --build build
```
### 2. Build Shared Library (`libtensor.so`)
```bash
# Navigate to Algorithm-Pyground/tensor
cmake -B build -S . -DBUILD_SHARED_LIBS=ON
cmake --build build
```

## Quick Start
### C++ Example
```c++
// example.cpp
#include <iostream>
#include <vector>
#include <assert.h>
#include "tensor/tensor.h"
#include "dtype/dtype.h"

using namespace std;

void print_2D_tensor(Tensor &x, string name) {
    assert(x.dim()==2);
    cout << "[" << name << "]\n";
    DISPATCH_BY_SCALAR_TYPE(x.scalar_type(), [&](){
        for (int i=0; i<x.size(0); ++i) {
            for (int j=0; j<x.size(1); ++j) {
                cout << x.at<cpp_scalar>({i, j}) <<", ";
            }
            cout << endl;
        }
    }())
}

void fill_2D_tensor(Tensor &x) {
    assert(x.dim()==2);
    DISPATCH_BY_SCALAR_TYPE(x.scalar_type(), [&](){
        for (int i=0; i<x.size(0); ++i) {
            for (int j=0; j<x.size(1); ++j) {
                x.at<cpp_scalar>({i, j}) = i*(x.size(0)+1)+j;
            }
        }
    }())
}

int main() {

    Tensor x_cpu({2, 3}, algo_pyground::FLOAT32, Device::CPU());
    fill_2D_tensor(x_cpu);

    Tensor y_cpu = x_cpu.to(algo_pyground::INT32);
    fill_2D_tensor(y_cpu);

    Tensor cpu_add_res = x_cpu + y_cpu;
    print_2D_tensor(cpu_add_res, "cpu_add_res");

    cout << endl;

    Tensor x_gpu = x_cpu.to(Device::CUDA());
    Tensor y_gpu = y_cpu.to(Device::CUDA());
    
    Tensor gpu_mul_res = x_gpu * y_gpu;
    gpu_mul_res = gpu_mul_res.to(Device::CPU());
    print_2D_tensor(gpu_mul_res, "gpu_mul_res");

    return 0;
}
```
### Python Interface
Python bindings via pybind11 are not yet implemented.

### Running the Example
if using `libtensor.so`
```shell
nvcc -Iinclude -I/usr/local/cuda/include example.cpp build/libtensor.so -L/usr/local/cuda/lib64 -lcudart -o example
LD_LIBRARY_PATH=build ./example
```

if using `libtensor.a`
```shell
nvcc -Iinclude -I/usr/local/cuda/include example.cpp build/libtensor.a -L/usr/local/cuda/lib64 -lcudart -o example
./example
```