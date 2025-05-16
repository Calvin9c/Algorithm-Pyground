#include <stdexcept>
#include "dtype/dtype.h"

#define INT8    ScalarType::INT8
#define INT16   ScalarType::INT16
#define INT32   ScalarType::INT32
#define INT64   ScalarType::INT64
#define UINT8   ScalarType::UINT8
#define UINT16  ScalarType::UINT16
#define UINT32  ScalarType::UINT32
#define UINT64  ScalarType::UINT64
#define FLOAT32 ScalarType::FLOAT32
#define FLOAT64 ScalarType::FLOAT64

inline constexpr int NUM_TYPES = 10;
inline constexpr ScalarType promote_table[NUM_TYPES][NUM_TYPES] = {
    //           INT8        INT16       INT32       INT64       UINT8      UINT16     UINT32     UINT64     FLOAT32    FLOAT64
    /*INT8*/    {INT8,       INT16,      INT32,      INT64,      INT16,     INT32,     INT64,     FLOAT32,   FLOAT32,   FLOAT64},
    /*INT16*/   {INT16,      INT16,      INT32,      INT64,      INT16,     INT32,     INT64,     FLOAT32,   FLOAT32,   FLOAT64},
    /*INT32*/   {INT32,      INT32,      INT32,      INT64,      INT32,     INT32,     INT64,     FLOAT32,   FLOAT32,   FLOAT64},
    /*INT64*/   {INT64,      INT64,      INT64,      INT64,      INT64,     INT64,     INT64,     FLOAT64,   FLOAT64,   FLOAT64},
    /*UINT8*/   {INT16,      INT16,      INT32,      INT64,      UINT8,     UINT16,    UINT32,    UINT64,    FLOAT32,   FLOAT64},
    /*UINT16*/  {INT32,      INT32,      INT32,      INT64,      UINT16,    UINT16,    UINT32,    UINT64,    FLOAT32,   FLOAT64},
    /*UINT32*/  {INT64,      INT64,      INT64,      INT64,      UINT32,    UINT32,    UINT32,    UINT64,    FLOAT32,   FLOAT64},
    /*UINT64*/  {FLOAT32,    FLOAT32,    FLOAT32,    FLOAT64,    UINT64,    UINT64,    UINT64,    UINT64,    FLOAT64,   FLOAT64},
    /*FLOAT32*/ {FLOAT32,    FLOAT32,    FLOAT32,    FLOAT64,    FLOAT32,   FLOAT32,   FLOAT32,   FLOAT64,   FLOAT32,   FLOAT64},
    /*FLOAT64*/ {FLOAT64,    FLOAT64,    FLOAT64,    FLOAT64,    FLOAT64,   FLOAT64,   FLOAT64,   FLOAT64,   FLOAT64,   FLOAT64}
};

#undef INT8
#undef INT16   
#undef INT32   
#undef INT64   
#undef UINT8   
#undef UINT16  
#undef UINT32  
#undef UINT64  
#undef FLOAT32 
#undef FLOAT64 

inline ScalarType promote_scalar_types(ScalarType a, ScalarType b) {
    int ai = static_cast<int>(a);
    int bi = static_cast<int>(b);
    if (ai < 0 || ai >= NUM_TYPES || bi < 0 || bi >= NUM_TYPES) {
        throw std::runtime_error("Invalid ScalarType in promote_types");
    }
    return promote_table[ai][bi];
}