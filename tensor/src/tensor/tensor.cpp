#include "tensor/tensor.h"
#include "dtype/dtype.h"
#include "device/device.h"
#include "tensor/tensor_op.h"
#include <iostream>

Tensor::Tensor():
_shape(), _dtype(DType()) {
    _storage = std::make_shared<Storage>(0, Device::CPU());
}

Tensor::Tensor(const std::vector<int64_t> &shape, DType dtype, Device device):
_shape(shape), _dtype(dtype) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty.");
    }
    int64_t num_elements = 1;
    for (const int64_t &s: shape) {
        if (s<=0) {
            throw std::invalid_argument("Tensor dimisions must be positive.");
        }
        num_elements *= s;
    }
    size_t nbytes = num_elements*dtype.itemsize();
    _storage = std::make_shared<Storage>(nbytes, device);
}

Tensor::Tensor(void *external_ptr, const std::vector<int64_t> &shape, DType dtype, Device device):
_shape(shape), _dtype(dtype) {
    if (!external_ptr) {
        throw std::invalid_argument("External pointer cannot be null.");
    }
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty.");
    }

    int64_t num_elements = 1;
    for (const int64_t &s: shape) {
        if (s<=0) {
            throw std::invalid_argument("Tensor dimension must be positive.");
        }
        num_elements *= s;
    }
    size_t nbytes = num_elements*dtype.itemsize();
    _storage = std::make_shared<Storage>(external_ptr, nbytes, device);
}

int64_t Tensor::numel() const {
    int64_t num_elements = 1;
    for (const int64_t &s: _shape) {
        num_elements *= s;
    }
    return !_shape.empty() ? num_elements : 0;
}

void Tensor::clear() {
    _storage->clear();
}

void Tensor::resize(const std::vector<int64_t> &new_shape) {

    int64_t new_num_elements = 1;
    for (const int64_t &s: new_shape) {
        if (s<=0) {
            throw std::invalid_argument("Resize shape must contain positive integers.");
        }
        new_num_elements *= s;
    }
    size_t new_nbytes = new_num_elements * _dtype.itemsize();

    _storage->resize(new_nbytes);
    _shape = new_shape;
}

Tensor Tensor::clone() const {
    Tensor res;
    res._shape = _shape;
    res._dtype = _dtype;
    res._storage = _storage->clone();
    return res;
}

Tensor Tensor::to(DType dtype) const {
    if (dtype==_dtype) {
        return clone();
    }
    Tensor res(_shape, dtype, device());
    tensor_op::cast(*this, res);
    return res;
}

Tensor Tensor::to(Device new_device) const {
    Tensor res;
    res._shape = _shape;
    res._dtype = _dtype;
    res._storage = _storage->to(new_device);
    return res;
}

Tensor Tensor::cpu() const {
    return to(Device::CPU());
}

Tensor Tensor::cuda(int index) const {
    return to(Device::CUDA(index));
}

Tensor Tensor::operator+(const Tensor &other) const {
    return tensor_op::add(*this, other);
}

Tensor Tensor::operator*(const Tensor &other) const {
    return tensor_op::mul(*this, other);
}

bool Tensor::operator==(const Tensor &other) const {
    return tensor_op::equal(*this, other);
}

Tensor Tensor::add(const Tensor &other) const {
    return (*this) + other;
}

Tensor Tensor::mul(const Tensor &other) const {
    return (*this) * other;
}

bool Tensor::equal(const Tensor &other) const {
    return (*this) == other;
}