cmake_minimum_required(VERSION 3.11)
project(TensorRT_plugin LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)

# 输出 TensorRT 配置路径
message(STATUS "TensorRT root directory: ${TensorRT_ROOT}")
message(STATUS "TensorRT include directory: ${TensorRT_INCLUDE_DIR}")
message(STATUS "TensorRT library directory: ${TensorRT_LIBRARIES}")

# 添加自定义插件库
add_library(customPlugin SHARED
    src/cpp/model.cpp
    src/cpp/utils.cpp
    src/cpp/customSELU_kernel.cu
    src/cpp/customSELU_plugin.cpp
    src/cpp/customScalar_kernel.cu
    src/cpp/customScalar_plugin.cpp
    src/cpp/customLeakyReLU_kernel.cu
    src/cpp/customLeakyReLU_plugin.cpp
)

# 设置头文件目录
target_include_directories(customPlugin PUBLIC "src/cpp")
target_include_directories(customPlugin PUBLIC "/usr/local/cuda/include")
target_include_directories(customPlugin PUBLIC ${OpenCV_INCLUDE_DIR})
target_include_directories(customPlugin PUBLIC ${TensorRT_INCLUDE_DIR})

# 链接 TensorRT 和 OpenCV 库
target_link_libraries(customPlugin PUBLIC
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
)

# 链接 CUDA 库
target_link_libraries(customPlugin PRIVATE
    ${CUDA_LIBRARIES}
)

set_target_properties(customPlugin PROPERTIES CUDA_ARCHITECTURES "61;70;75")

# 添加可执行文件
add_executable(main
    src/cpp/main.cpp
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
    ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)
target_include_directories(main PRIVATE ${TensorRT_INCLUDE_DIR} ${TensorRT_SAMPLE_DIR}/common)

# 链接自定义插件库
target_link_libraries(main PRIVATE
    customPlugin
    ${CUDA_LIBRARIES}
    stdc++fs
)

set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "61;70;75")