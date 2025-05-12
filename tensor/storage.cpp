#include "mem_util.h"
#include "storage.h"
#include "dtype.h"
#include <stdexcept>
#include <cstring>

// --------------------
// Constructor
// --------------------
Storage::Storage(size_t nbytes, const Device &device):
_nbytes(nbytes), _device(device), _owning(true) {
    _allocate();
}

Storage::Storage(void *external_ptr, size_t nbytes, const Device &device): 
_data(external_ptr), _nbytes(nbytes), _device(device), _owning(false) {
    if (!_data) {
        throw std::invalid_argument("Cannot wrap null external pointer");
    }
}

// --------------------
// Move Constructor & Assignment Operator
// --------------------
Storage::Storage(Storage &&other) noexcept:
_data(other._data), _nbytes(other._nbytes),
_device(other._device), _owning(other._owning) {
    other._data = nullptr;
    other._nbytes = 0;
    other._owning = false;
}

Storage& Storage::operator=(Storage &&other) noexcept {
    if (this != &other) {
        _deallocate();
        _data = other._data;
        _nbytes = other._nbytes;
        _device = other._device;
        _owning = other._owning;

        other._data = nullptr;
        other._nbytes = 0;
        other._owning = false;
    }
    return *this;
}

// --------------------
// Destructor
// --------------------
Storage::~Storage(){
    _deallocate();
}

std::shared_ptr<Storage> Storage::clone() const {
    /*
        Storage *raw = new Storage(nbytes, device);
        shared_ptr<Storage> cloned(raw);
    */
    auto cloned = std::make_shared<Storage>(_nbytes, _device);
    _memcpy(cloned->_data, cloned->_device, _data, _device, _nbytes);
    return cloned;
}

std::shared_ptr<Storage> Storage::to(const Device &new_device) const {
    if (new_device == _device) return clone();
    auto res = std::make_shared<Storage>(_nbytes, new_device);
    _memcpy(res->_data, new_device, _data, _device, _nbytes);
    return res;
}

void Storage::resize(size_t nbytes) {
    if (nbytes<=_nbytes) return;
    
    void *new_data = _malloc(nbytes, _device);
    _memcpy(new_data, _device, _data, _device, _nbytes);
    _memfree(_data, _device);

    _data = new_data;
    _nbytes = nbytes;
}

void Storage::clear() {
    _memset(_data, 0, _nbytes, _device);
}

void Storage::_allocate() {
    _data = _malloc(_nbytes, _device);
}

void Storage::_deallocate() {
    if (_owning&&_data) {
        _memfree(_data, _device);
        _data = nullptr;
        _nbytes = 0;
    }
}

#define OP(scalar_type, cpp_scalar, ...) \
template cpp_scalar* Storage::data_as<cpp_scalar>() const;
FOR_EACH_SCALAR_TYPE(OP)
#undef OP