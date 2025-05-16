#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include "device/device.h"
#include "dtype/dtype.h"
#include "mem/mem_util.h"

// type-erased raw memory
class Storage {
public:
    // Constructor: Allocate new memory
    Storage(size_t nbytes, const Device& device);

    // Constructor: Wrap external memory (non-owning)
    Storage(void* external_ptr, size_t nbytes, const Device& device);

    // Delete Copy Constructor
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Move Constructor
    Storage(Storage &&other) noexcept;
    Storage& operator=(Storage &&other) noexcept;

    // Destructor: Free memory if owning
    ~Storage();

    // Clone the entire memory block (deep copy)
    std::shared_ptr<Storage> clone() const;
    std::shared_ptr<Storage> to(const Device &new_device) const;

    // modify memory space
    void resize(size_t nbytes);
    void clear();
    
    // Accessors
    void* data() const { return _data; }

    template <typename Scalar>
    Scalar* data_as() const { return static_cast<Scalar*>(_data); }

    size_t nbytes() const { return _nbytes; }
    const Device& device() const { return _device; }
    bool owns_memory() const { return _owning; }

private:
    void *_data = nullptr;
    size_t _nbytes = 0;
    Device _device;
    bool _owning = true;

    // Internal allocation and free logic
    void _allocate();
    void _deallocate();
};
