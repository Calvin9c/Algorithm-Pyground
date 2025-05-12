#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>
#include <type_traits>

// --------------------------------------
// SUPPORT_SCALARS 
// --------------------------------------
#define FOR_EACH_SCALAR_TYPE(MACRO, ...) \
    MACRO(INT8,    int8_t, __VA_ARGS__) \
    MACRO(INT16,   int16_t, __VA_ARGS__) \
    MACRO(INT32,   int32_t, __VA_ARGS__) \
    MACRO(INT64,   int64_t, __VA_ARGS__) \
    MACRO(UINT8,   uint8_t, __VA_ARGS__) \
    MACRO(UINT16,  uint16_t, __VA_ARGS__) \
    MACRO(UINT32,  uint32_t, __VA_ARGS__) \
    MACRO(UINT64,  uint64_t, __VA_ARGS__) \
    MACRO(FLOAT32, float, __VA_ARGS__) \
    MACRO(FLOAT64, double, __VA_ARGS__)

#define UNKNOWN_SCALAR(MACRO, ...) \
    MACRO(UNKNOWN, void, __VA_ARGS__)

#define SWITCH_CASE(SCALAR_TYPE, CPP_SCALAR, ...) \
    case ScalarType::SCALAR_TYPE: { \
        using cpp_scalar = CPP_SCALAR; \
        __VA_ARGS__; \
        break; \
    }


#define DISPATCH_BY_SCALAR_TYPE(SCALAR_TYPE, ...) \
    switch (SCALAR_TYPE) { \
        FOR_EACH_SCALAR_TYPE(SWITCH_CASE, __VA_ARGS__) \
        default: \
            throw std::runtime_error("Unsupported ScalarType in dispatch."); \
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
// DType class
// --------------------------------------
class DType {
public:
    DType() : _type(ScalarType::UNKNOWN) {}
    explicit DType(ScalarType type) : _type(type) {}

    ScalarType type() const { return _type; }

    std::string name() const {
        switch (_type) {
#define GET_STR(SCALAR_TYPE, CPP_SCALAR, ...) \
    case ScalarType::SCALAR_TYPE: { \
        return #SCALAR_TYPE; \
    }
    FOR_EACH_SCALAR_TYPE(GET_STR)
    UNKNOWN_SCALAR(GET_STR)
#undef GET_STR
            default: return "unknown";
        }
    }

    size_t itemsize() const {
        switch (_type) {
#define SIZE_OF(SCALAR_TYPE, CPP_SCALAR, ...) \
    case ScalarType::SCALAR_TYPE: { \
        return sizeof(CPP_SCALAR); \
    };
    FOR_EACH_SCALAR_TYPE(SIZE_OF)
#undef SIZE_OF
            default: throw std::runtime_error("Unknown dtype.");
        }
    }

    bool operator==(const DType& other) const {
        return _type == other._type;
    }

    bool operator!=(const DType& other) const {
        return !(*this == other);
    }

    template <typename Scalar>
    static DType from();

private:
ScalarType _type;
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

template <typename Scalar>
DType DType::from() {
    return DType(DTypeTrait<Scalar>::type);
}