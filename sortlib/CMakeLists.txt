cmake_minimum_required(VERSION 3.15)
project(sortlib)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(pybind11 REQUIRED)

include_directories(cpp/include)

pybind11_add_module(sortlib_native cpp/src/pybind_wrapper.cpp)

set_target_properties(sortlib_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)