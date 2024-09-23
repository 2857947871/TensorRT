cmake_minimum_required(VERSION 3.11)
project(TensorRT_plugin LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)

# 查找 CUDA
find_package(CUDA REQUIRED)

# 输出 TensorRT 配置路径
message(STATUS "TensorRT root directory: ${TensorRT_ROOT}")
message(STATUS "TensorRT include directory: ${TensorRT_INCLUDE_DIR}")
message(STATUS "TensorRT library directory: ${TensorRT_LIBRARIES}")

# 设置包含目录
include_directories(/usr/local/cuda/include)
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${TensorRT_INCLUDE_DIR})
include_directories(${TensorRT_SAMPLE_DIR}/common)


# 添加可执行文件
add_executable(build
    src/cpp/build.cpp
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
    ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)

target_link_libraries(build PRIVATE
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    stdc++fs
)

set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")















