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