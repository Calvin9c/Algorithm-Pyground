cmake_minimum_required(VERSION 3.15)
project(linked_list)

set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 REQUIRED)

set(INCLUDE_DIR cpp/include)
set(SRC_DIR cpp/src)

add_library(linked_list_native MODULE
    ${SRC_DIR}/pybind_wrapper.cpp
)
target_include_directories(linked_list_native PRIVATE ${INCLUDE_DIR})
target_link_libraries(linked_list_native PRIVATE pybind11::module)

set_target_properties(linked_list_native PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)