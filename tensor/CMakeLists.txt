cmake_minimum_required(VERSION 3.18)
project(tensorlib LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TENSOR_SRC
    src/tensor/tensor.cpp
    src/tensor/tensor_op.cpp
    src/tensor/tensor_kernel.cpp
    src/tensor/tensor_kernel.cu
)

set(STORAGE_SRC
    src/storage/storage.cpp
)

set(MEM_SRC
    src/mem/mem_util.cpp
)

set(SOURCES
    ${TENSOR_SRC}
    ${STORAGE_SRC}
    ${MEM_SRC}
)

option(BUILD_SHARED_LIBS
    "Build shared libraries instead of static ones" OFF
)

add_library(tensor ${SOURCES})

set_target_properties(tensor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;86"
)

target_include_directories(tensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(tensor
    PUBLIC 
        CUDA::cudart
        OpenMP::OpenMP_CXX
)

enable_testing()
file(GLOB TEST_FILES tests/test_*.cpp)
foreach(f IN LISTS TEST_FILES)
    get_filename_component(test ${f} NAME_WE)
    add_executable(${test} ${f})
    target_link_libraries(${test} tensor)
    add_test(NAME ${test} COMMAND ${test})
endforeach()