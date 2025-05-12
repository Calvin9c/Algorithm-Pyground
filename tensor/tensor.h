#pragma once

#include <vector>
#include <stdexcept>
#include "dtype.h"
#include "device.h"
#include "storage.h"

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
    int64_t dim() const {return static_cast<int64_t>(_shape.size());}

    const std::vector<int64_t>& sizes() const { return _shape; }
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
    const Device& device() const {return _storage->device();}

    // data access
    void* data() const {return _storage->data();}
    
    template <typename Scalar>
    Scalar* data_as() const {return _storage->data_as<Scalar>();}

    size_t nbytes() const {return _storage->nbytes();}
    bool empty() const {return _storage->nbytes()==0;}

    void clear();
    void resize(const std::vector<int64_t> &new_shape);
    std::shared_ptr<Tensor> clone() const;

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
    std::vector<int64_t> _shape;
    DType _dtype;
    std::shared_ptr<Storage> _storage;
};
