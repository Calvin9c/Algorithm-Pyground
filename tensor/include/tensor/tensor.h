#pragma once

#include <vector>
#include <stdexcept>
#include "dtype/dtype.h"
#include "device/device.h"
#include "storage/storage.h"

class Tensor {
public:
    // constructor
    Tensor();
    Tensor(const std::vector<int64_t> &shape, DType dtype, Device device=Device::CPU());
    Tensor(void *external_ptr, const std::vector<int64_t> &shape, DType dtype, Device device);
    // copy constructor
    Tensor(const Tensor&) = default;
    // copy assignment operator
    Tensor& operator=(const Tensor&) = default;
    // move constructor
    Tensor(Tensor&&) noexcept = default;
    // move assignment operator
    Tensor& operator=(Tensor&&) noexcept = default;
    // destructor
    ~Tensor() = default;

    // basic info
    const std::vector<int64_t>& shape() const {return _shape;}
    std::vector<int64_t>& shape() {return _shape;}
    const std::vector<int64_t>& sizes() const { return _shape; }
    std::vector<int64_t>& sizes() { return _shape; }
    const std::vector<int64_t>& stride() const {return _stride;}
    std::vector<int64_t>& stride() {return _stride;}

    int64_t dim() const {return static_cast<int64_t>(_shape.size());}
    int64_t size(int i) const {
        int64_t dim = this->dim();
        if (i<0) i+=dim;
        if (i<0 || i>=dim){
            throw std::out_of_range("Tensor::size(i): index out of bounds");
        }
        return _shape[i];
    }
    int64_t numel() const;

    DType dtype() const {return _dtype;}
    ScalarType scalar_type() const {return _dtype.type();}
    const Device& device() const {return _storage->device();}
    DeviceType device_type() const {return (_storage->device()).type();}

    // data access
    void* data() const {return _storage->data();}
    
    template <typename Scalar>
    Scalar* data_as() const {
        if (!_dtype.is<Scalar>()) {
            throw std::runtime_error("Tensor::data_as(): dtype mismatch");
        }
        return _storage->data_as<Scalar>();
    }

    size_t nbytes() const {return _storage->nbytes();}
    bool empty() const {return _storage->nbytes()==0;}

    void clear();
    void resize(const std::vector<int64_t> &new_shape);
    Tensor clone() const;

    template<typename Scalar>
    Scalar item() const {
        if (!_dtype.is<Scalar>()) {
            throw std::runtime_error("Tensor::item(): dtype mismatch");
        }
        if (numel()!=1) {
            throw std::runtime_error("Tensor::item() requires numel == 1");
        }
        return data_as<Scalar>()[0];
    }

    std::vector<int64_t> init_stride(const std::vector<int64_t> &shape) {
        std::vector<int64_t>res(shape.size(), 0);
        int64_t s=1;
        for (int64_t i=dim()-1; i>=0; --i) {
            res[i] = s;
            s *= shape[i];
        }
        return res;
    }

    template<typename Scalar>
    Scalar& at(const std::vector<int64_t> &index) {
        if (!_dtype.is<Scalar>()) {
            throw std::runtime_error("Tensor::at(): dtype mismatch");
        }
        if (index.size() != _shape.size()) {
            throw std::runtime_error("Tensor::at(): index dimension mismatch");
        }

        const int N = index.size();
        int64_t flat_i = 0;
        for (int i=N-1; i>=0; --i) {
            if (index[i]>=_shape[i] || index[i]<0) {
                throw std::out_of_range("Tensor::at(): index out of bounds");
            }
            flat_i += index[i] * _stride[i];
        }
        return data_as<Scalar>()[flat_i];
    }

    Tensor to(DType dtype) const;
    Tensor to(Device new_device) const;
    Tensor cpu() const;
    Tensor cuda(int index=0) const;

    Tensor add(const Tensor &other) const;
    Tensor mul(const Tensor &other) const;
    bool equal(const Tensor &other) const;

    Tensor operator+(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;
    bool operator==(const Tensor &other) const;
   
private:
    std::vector<int64_t> _shape, _stride;
    DType _dtype;
    std::shared_ptr<Storage> _storage;
};
