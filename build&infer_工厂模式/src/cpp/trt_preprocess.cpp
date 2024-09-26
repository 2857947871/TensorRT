# include "utils.hpp"
# include "trt_timer.hpp"
# include "trt_preprocess.hpp"
# include "opencv2/opencv.hpp"

namespace process {

// 根据scale进行缩放
cv::Mat preprocess_resize_cpu(const cv::Mat &src, const int &tarH, const int &tarW, \
            float* mean, float& std, tactics tac)
{
    cv::Mat tar;
    int height  = src.rows;
    int width   = src.cols;
    int dim     = std::max(height, width);

    // BGR2RGB
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);

    // Resize
    switch (tac)
    {
    case tactics::CPU_NEAREST:
            cv::resize(src, tar, cv::Size(tarW, tarH), 0, 0, cv::INTER_NEAREST);
            break;
    case tactics::CPU_BILINEAR:
            cv::resize(src, tar, cv::Size(tarW, tarH), 0, 0, cv::INTER_LINEAR);
            break;
    default:
        LOGE("ERROR: Wrong CPU resize tactics selected. Program terminated");
        exit(1);
    }

    return tar;
}

void preprocess_resize_gpu(const cv::Mat &h_src, float* d_tar, const int &tarH, const int &tarW, \
            float* h_mean, float* h_std, tactics tac)
{
    float*   d_mean = nullptr;
    float*   d_std  = nullptr;
    uint8_t* d_src  = nullptr;

    int height  = h_src.rows;
    int width   = h_src.cols;
    int channel = 3;

    int src_size  = height * width * channel * sizeof(uint8_t);
    int norm_size = 3 * sizeof(float);

    // 分配内存并拷贝
    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    CUDA_CHECK(cudaMalloc(&d_mean, norm_size));
    CUDA_CHECK(cudaMalloc(&d_std, norm_size));

    CUDA_CHECK(cudaMemcpy(d_src, h_src.data, src_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean, h_mean, norm_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_std, h_std, norm_size, cudaMemcpyHostToDevice));

    // kernel function
    resize_gpu(d_src, d_tar, width, height, tarW, tarH, d_mean, d_std, tac);

    // host和device进行同步处理
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_std));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_src));

    // 接下来的推理也在device端, 因此不返回
}























}; // namespace process 与.hpp文件对齐