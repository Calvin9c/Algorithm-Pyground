#pragma once

#include <string>
#include <sstream>
#include <stdexcept>

enum class DeviceType {
    CPU, CUDA
};

class Device {
    public:
        explicit Device(DeviceType type=DeviceType::CPU, int index=0) : _type(type), _index(index) {
            if (_type == DeviceType::CPU && _index != 0) {
                throw std::invalid_argument("CPU device must have index 0.");
            }
        }

        static Device CPU() {
            return Device(DeviceType::CPU, 0);
        }
    
        static Device CUDA(int index = 0) {
            return Device(DeviceType::CUDA, index);
        }

        bool is_cpu() const {
            return _type == DeviceType::CPU;
        }

        bool is_cuda() const {
            return _type == DeviceType::CUDA;
        }

        DeviceType type() const {return _type;}

        int index() const {return _index;}

        bool operator==(const Device& other) const {
            return _type == other._type;
        }

        bool operator!=(const Device& other) const {
            return !(*this==other);
        }

        std::string str() const {
            if (_type == DeviceType::CPU) {
                return "cpu";
            } else { // _type == DeviceType::CUDA
                return "cuda:" + std::to_string(_index);
            }
        }

    private:
        DeviceType _type;
        int _index;
};

inline std::ostream& operator<<(std::ostream& os, const Device& d) {
    return os << d.str();
}