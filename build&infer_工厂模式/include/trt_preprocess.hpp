# ifndef __PREPROCESS_HPP__
# define __PREPROCESS_HPP__

# include "opencv2/opencv.hpp"
# include "trt_timer.hpp"

namespace process {

enum class tactics : int32_t {
    CPU_NEAREST         = 0,
    CPU_BILINEAR        = 1,
    GPU_NEAREST         = 2,
    GPU_NEAREST_CENTER  = 3,
    GPU_BILINEAR        = 4,
    GPU_BILINEAR_CENTER = 5
};

cv::Mat preprocess_resize_cpu(const cv::Mat &src, const int &tarH, const int &tarW, float* mean, float& std, tactics tac);
void    preprocess_resize_gpu(const cv::Mat &h_src, float* d_tar, const int &tarH, const int &tarW, float* h_mean, float* h_std, tactics tac);
void    resize_gpu(const uint8_t* d_src, float* d_tar, int srcW, int srcH, int tarW, int tarH, float* d_mean, float* d_std, process::tactics tac);
}; // namespace process
# endif