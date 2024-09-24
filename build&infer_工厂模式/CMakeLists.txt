# 项目基本信息
cmake_minimum_required(VERSION 3.11)
project(CNN_Deploy LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

# 必要的函数库
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)

# 输出配置路径
message(STATUS "TensorRT root directory:    ${TensorRT_ROOT}")
message(STATUS "TensorRT library directory: ${TensorRT_LIBRARIES}")
message(STATUS "TensorRT include directory: ${TensorRT_INCLUDE_DIR}")

# 头文件
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)
include_directories(${TensorRT_INCLUDE_DIR})

# 添加可执行文件
add_executable(main
    src/cpp/main.cpp
    src/cpp/utils.cpp
    src/cpp/trt_model.cpp
    src/cpp/trt_timer.cpp
    src/cpp/trt_logger.cpp
    src/cpp/trt_worker.cpp
    src/cpp/trt_preprocess.cu
    src/cpp/trt_preprocess.cpp
    src/cpp/trt_classifier.cpp
)

# 链接库
target_link_libraries(main
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    /usr/local/cuda/lib64/libcudart.so  # 添加 CUDA 运行时库
    stdc++fs
)