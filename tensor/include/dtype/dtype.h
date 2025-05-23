#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>
#include <type_traits>
#include "device/device.h"

// --------------------------------------
// SUPPORT_SCALARS 
// --------------------------------------
#define FOR_EACH_SCALAR_TYPE(MACRO, ...)  \
    MACRO(INT8,    int8_t,   __VA_ARGS__) \
    MACRO(INT16,   int16_t,  __VA_ARGS__) \
    MACRO(INT32,   int32_t,  __VA_ARGS__) \
    MACRO(INT64,   int64_t,  __VA_ARGS__) \
    MACRO(UINT8,   uint8_t,  __VA_ARGS__) \
    MACRO(UINT16,  uint16_t, __VA_ARGS__) \
    MACRO(UINT32,  uint32_t, __VA_ARGS__) \
    MACRO(UINT64,  uint64_t, __VA_ARGS__) \
    MACRO(FLOAT32, float,    __VA_ARGS__) \
    MACRO(FLOAT64, double,   __VA_ARGS__)

#define UNKNOWN_SCALAR(MACRO, ...) \
    MACRO(UNKNOWN, void, __VA_ARGS__)

#define EXPAND_WITH_DST(SRC_ENUM, SRC_CPP, MACRO) \
    MACRO(SRC_ENUM, SRC_CPP, INT8,    int8_t  )   \
    MACRO(SRC_ENUM, SRC_CPP, INT16,   int16_t )   \
    MACRO(SRC_ENUM, SRC_CPP, INT32,   int32_t )   \
    MACRO(SRC_ENUM, SRC_CPP, INT64,   int64_t )   \
    MACRO(SRC_ENUM, SRC_CPP, UINT8,   uint8_t )   \
    MACRO(SRC_ENUM, SRC_CPP, UINT16,  uint16_t)   \
    MACRO(SRC_ENUM, SRC_CPP, UINT32,  uint32_t)   \
    MACRO(SRC_ENUM, SRC_CPP, UINT64,  uint64_t)   \
    MACRO(SRC_ENUM, SRC_CPP, FLOAT32, float   )   \
    MACRO(SRC_ENUM, SRC_CPP, FLOAT64, double  )

#define FOR_EACH_SCALAR_TYPE_PAIR(MACRO) \
    FOR_EACH_SCALAR_TYPE(EXPAND_WITH_DST, MACRO)

#define BUILD_SWITCH_CASE(SCALAR_TYPE, CPP_SCALAR, ...) \
    case ScalarType::SCALAR_TYPE: { \
        using cpp_scalar = CPP_SCALAR; \
        __VA_ARGS__; \
        break; \
    }

#define DISPATCH_BY_SCALAR_TYPE(SCALAR_TYPE, ...) \
    switch (SCALAR_TYPE) { \
        FOR_EACH_SCALAR_TYPE(BUILD_SWITCH_CASE, __VA_ARGS__) \
        default: \
            throw std::runtime_error("Unsupported ScalarType in dispatch."); \
    }

#define MATMUL_SCALAR_TYPE(MACRO, ...) \
    MACRO(FLOAT32, float,  __VA_ARGS__) \
    MACRO(FLOAT64, double, __VA_ARGS__)

#define MATMUL_KERNEL(SCALAR_TYPE, FN) \
    switch (SCALAR_TYPE) { \
        MATMUL_SCALAR_TYPE(BUILD_SWITCH_CASE, FN) \
        default: \
            throw std::runtime_error("Unsupported ScalarType in dispatch."); \
    }

#define DISPATCH_TO_KERNEL(DEVICE_TYPE, SCALAR_TYPE, KERNEL, LAUNCHER, ...) \
    switch (DEVICE_TYPE) { \
        case DeviceType::CPU: { \
            KERNEL(SCALAR_TYPE, tensor_kernel::cpu::LAUNCHER<cpp_scalar>(__VA_ARGS__)) \
            break; \
        } \
        case DeviceType::CUDA: { \
            KERNEL(SCALAR_TYPE, tensor_kernel::cuda::LAUNCHER<cpp_scalar>(__VA_ARGS__)) \
            break; \
        } \
        default: \
            throw std::runtime_error("Unsupported Backend in dispatch."); \
    }

// --------------------------------------
// ScalarType enum
// --------------------------------------
enum class ScalarType {
#define GET_SCALAR_TYPE(SCALAR_TYPE, CPP_SCALAR, ...) SCALAR_TYPE,
    FOR_EACH_SCALAR_TYPE(GET_SCALAR_TYPE)
    UNKNOWN_SCALAR(GET_SCALAR_TYPE)
#undef GET_SCALAR_TYPE
};

// --------------------------------------
// DTypeTrait
// --------------------------------------
template <typename Scalar>
struct DTypeTrait;

#define OP(scalar_type, cpp_scalar, ...)                        \
template <>                                              \
struct DTypeTrait<cpp_scalar> {                          \
    static constexpr ScalarType type = ScalarType::scalar_type; \
};
    FOR_EACH_SCALAR_TYPE(OP)
    UNKNOWN_SCALAR(OP)
#undef OP

// --------------------------------------
// DType class
// --------------------------------------
class DType {
public:
    DType() : _type(ScalarType::UNKNOWN) {}
    constexpr explicit DType(ScalarType type) : _type(type) {}

    ScalarType type() const { return _type; }

    template<typename Scalar>
    bool is() const {
        return *this == DType::from<Scalar>();
    }

    std::string name() const {
        switch (_type) {
#define GET_STR(SCALAR_TYPE, CPP_SCALAR, ...) \
    case ScalarType::SCALAR_TYPE: { \
        return #SCALAR_TYPE; \
    }
    FOR_EACH_SCALAR_TYPE(GET_STR)
#undef GET_STR
            default: return "unknown";
        }
    }

    size_t itemsize() const {
        DISPATCH_BY_SCALAR_TYPE(_type, [&](){
            return sizeof(cpp_scalar);
        }())
    }

    bool operator==(const DType& other) const {
        return _type == other._type;
    }

    bool operator!=(const DType& other) const {
        return !(*this == other);
    }

    template <typename Scalar>
    static DType from() {
        return DType(DTypeTrait<Scalar>::type);
    };

    #define STATIC_DTYPE_FN(SCALAR_TYPE, CPP_SCALAR, ...) \
        constexpr static DType SCALAR_TYPE() { \
            return DType(ScalarType::SCALAR_TYPE); \
        } 
    FOR_EACH_SCALAR_TYPE(STATIC_DTYPE_FN)
    #undef STATIC_DTYPE_FN

private:
ScalarType _type;
};

namespace algo_pyground {
    // 仿照 torch 對於 C++17 以下版本的支援
    // 不使用 inline constexpr
    #define DECLARE_DTYPE_OBJ(SCALAR_TYPE, CPP_SCALAR, ...) \
        constexpr DType SCALAR_TYPE = DType::SCALAR_TYPE();
    FOR_EACH_SCALAR_TYPE(DECLARE_DTYPE_OBJ)
    #undef DECLARE_DTYPE_OBJ
}