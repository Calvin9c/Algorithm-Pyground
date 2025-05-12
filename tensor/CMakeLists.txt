cmake_minimum_required(VERSION 3.18)
project(tensorlib LANGUAGES C CXX CUDA)

# 設定編譯器標準與 flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 啟用 position independent code（支援 .so/.a）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# === 所有源碼 ===
set(SOURCES
    device.h
    dtype.h
    mem_util.cpp
    mem_util.h
    storage.cpp
    storage.h
    tensor.cpp
    tensor.h
    tensor_op.cpp
    tensor_op.h
    tensor_kernel.cpp
    tensor_kernel.cu
    tensor_kernel.h
)

# === 建立靜態庫 ===
add_library(tensorlib STATIC ${SOURCES})

# === 對 CUDA 檔案使用不同 flags（可選）===
set_target_properties(tensorlib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;86"
)

# === 包含目錄 ===
target_include_directories(tensorlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# === CUDA 依賴 ===
find_package(CUDAToolkit REQUIRED)
target_link_libraries(tensorlib PUBLIC CUDA::cudart)
