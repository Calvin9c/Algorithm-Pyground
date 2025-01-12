cmake_minimum_required(VERSION 3.18)
project(image_processing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find pybind11
find_package(pybind11 REQUIRED)

# CUDA 
enable_language(CUDA)
set(CUDA_ARCHITECTURES 52 60 70 75 80)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# set variable `SOURCE_FILES`, and add CUDA, C++ file
set(SOURCE_FILES
    cpp/src/pybind_wrapper.cpp
    cpp/src/convolution.cu
)
# header file
include_directories(cpp/include)

# named shared lib
set(MODULE_NAME image_processing_native)

# build the target
add_library(${MODULE_NAME} SHARED ${SOURCE_FILES})
# target <--- CUDA
set_target_properties(${MODULE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target <--- pybind11
target_link_libraries(${MODULE_NAME} PRIVATE pybind11::module)

# compile option
target_compile_options(${MODULE_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

set_target_properties(${MODULE_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
