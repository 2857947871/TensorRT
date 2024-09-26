# include <random>
# include <fstream>
# include <iostream>
# include <algorithm>
# include <opencv2/core/core.hpp>
# include <opencv2/highgui/highgui.hpp>
# include "utils.hpp"
# include "trt_model.hpp"
# include "trt_logger.hpp"
# include "trt_preprocess.hpp"
# include "trt_calibrator.hpp"


namespace model
{
# if 1
// Entroy
Int8EntropyCalibrator::Int8EntropyCalibrator(
        const int &batchSize, 
        const std::string &calibrationDataPath,
        const std::string &calibrationTablePath,
        const int& inputSize,
        const int& inputH, const int& inputW) :
        m_batchSize(batchSize),
        m_inputH(inputH), m_inputW(inputW),
        m_calibrationTablePath(calibrationTablePath),
        m_inputSize(inputSize), m_inputCount(batchSize * inputSize)
{
    // 确保是batch的整数倍并打乱顺序
    m_imageList = loadDataList(calibrationDataPath);
    m_imageList.resize(static_cast<int>((m_imageList.size() / m_batchSize) * m_batchSize));
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(m_imageList.begin(), m_imageList.end(), g);
    // std::random_shuffle(m_imageList.begin(), m_imageList.end(),
    //                     [](int i){ return rand() % i; });

    CUDA_CHECK(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));
}

// 加载一个batch
bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    
    if (m_imageIndex + m_batchSize >= m_imageList.size() + 1)
        return false;
    
    LOG("%3d/%3d (%3dx%3d): %s", 
        m_imageIndex + 1, m_imageList.size(), m_inputH, m_inputW, m_imageList.at(m_imageIndex).c_str());
    
    // 前处理(与inference阶段保持一致)
    cv::Mat input_image;
    float mean[] = {0.406, 0.456, 0.485};
    float std[]  = {0.225, 0.224, 0.229};
    for (int i = 0; i < m_batchSize; i ++){
        input_image = cv::imread(m_imageList.at(m_imageIndex++));
        process::preprocess_resize_gpu(
            input_image, 
            m_deviceInput + i * m_inputSize,
            m_inputH, m_inputW, 
            mean, std, process::tactics::GPU_BILINEAR);
    }

    bindings[0] = m_deviceInput;

    return true;
}

// 读取calibration table来创建INT8的engine
const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept {
    
    void* output;
    m_calibrationCache.clear();

    // 以二进制方式打开table
    std::ifstream input(m_calibrationTablePath, std::ios::binary);
    input >> std::noskipws; // 忽略空白字符, 确保完整读取二进制文件
    if (m_readCache && input.good()) // input.good(): 文件是否打开成功
        std::copy(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>(), std::back_inserter(m_calibrationCache));

    length = m_calibrationCache.size();
    if (length){
        LOG("Using cached calibration table to build INT8 trt engine...");
        output = &m_calibrationCache[0];
    }else{
        LOG("Creating new calibration table to build INT8 trt engine...");
        output = nullptr;
    }
    return output;
}

// 将calibration cache写入table
void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::ofstream output(m_calibrationTablePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}
# endif

# if 1
// MinMax
Int8MinMaxCalibrator::Int8MinMaxCalibrator(
        const int &batchSize, 
        const std::string &calibrationDataPath,
        const std::string &calibrationTablePath,
        const int& inputSize,
        const int& inputH, const int& inputW) :
        m_batchSize(batchSize),
        m_inputH(inputH), m_inputW(inputW),
        m_calibrationTablePath(calibrationTablePath),
        m_inputSize(inputSize), m_inputCount(batchSize * inputSize)
{
    // 确保是batch的整数倍并打乱顺序
    m_imageList = loadDataList(calibrationDataPath);
    m_imageList.resize(static_cast<int>((m_imageList.size() / m_batchSize) * m_batchSize));
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(m_imageList.begin(), m_imageList.end(), g);
    // std::random_shuffle(m_imageList.begin(), m_imageList.end(),
    //                     [](int i){ return rand() % i; });

    CUDA_CHECK(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));
}

// 加载一个batch
bool Int8MinMaxCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    
    if (m_imageIndex + m_batchSize >= m_imageList.size() + 1)
        return false;
    
    LOG("%3d/%3d (%3dx%3d): %s", 
        m_imageIndex + 1, m_imageList.size(), m_inputH, m_inputW, m_imageList.at(m_imageIndex).c_str());
    
    // 前处理(与inference阶段保持一致)
    cv::Mat input_image;
    float mean[] = {0.406, 0.456, 0.485};
    float std[]  = {0.225, 0.224, 0.229};
    for (int i = 0; i < m_batchSize; i ++){
        input_image = cv::imread(m_imageList.at(m_imageIndex++));
        process::preprocess_resize_gpu(
            input_image, 
            m_deviceInput + i * m_inputSize,
            m_inputH, m_inputW, 
            mean, std, process::tactics::GPU_BILINEAR);
    }

    bindings[0] = m_deviceInput;

    return true;
}

// 读取calibration table来创建INT8的engine
const void* Int8MinMaxCalibrator::readCalibrationCache(size_t& length) noexcept {
    
    void* output;
    m_calibrationCache.clear();

    // 以二进制方式打开table
    std::ifstream input(m_calibrationTablePath, std::ios::binary);
    input >> std::noskipws; // 忽略空白字符, 确保完整读取二进制文件
    if (m_readCache && input.good()) // input.good(): 文件是否打开成功
        std::copy(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>(), std::back_inserter(m_calibrationCache));

    length = m_calibrationCache.size();
    if (length){
        LOG("Using cached calibration table to build INT8 trt engine...");
        output = &m_calibrationCache[0];
    }else{
        LOG("Creating new calibration table to build INT8 trt engine...");
        output = nullptr;
    }
    return output;
}

// 将calibration cache写入table
void Int8MinMaxCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::ofstream output(m_calibrationTablePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}
# endif
}; // namespace model
