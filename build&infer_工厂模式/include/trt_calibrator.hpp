# ifndef __TRT_CALIBRATOR_HPP__
# define __TRT_CALIBRATOR_HPP__

# include <vector>
# include <string>
# include <iostream>
# include "NvInfer.h"


namespace model
{
// 自定义一个calibrator类, 继承自nvinfer1中的calibrator
// TensorRT有五种量化方式
//  1. nvinfer1::IInt8EntropyCalibrator
//      熵最小化的方式来选择量化边界
//  2. nvinfer1::IInt8MinMaxCalibrator
//      每一层的最大最小值来确定量化范围, 如果分布不均匀 -> 误差较大
//  3. nvinfer1::IInt8EntropyCalibrator2
//      1的升级版, TensorRT的默认calibrator
//  4. nvinfer1::IInt8LegacyCalibrator
//      早期的TensortRT calibrator, 不推荐
//  5. nvinfer1::IInt8Calibrator
//      抽象类, 1234均由此继承

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator(
        const int &batchSize, 
        const std::string &calibrationDataPath,
        const std::string &calibrationTablePath,
        const int& inputSize,
        const int& inputH, const int& inputW);
    
    ~Int8EntropyCalibrator() {};
    
public:
    int         getBatchSize() const noexcept override {return m_batchSize;}
    bool        getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(std::size_t &length) noexcept override;
    void        writeCalibrationCache (const void* ptr, std::size_t legth) noexcept override;

private:
    const int m_batchSize;
    const int m_inputH;
    const int m_inputW;
    const int m_inputSize;
    const int m_inputCount;
    const std::string m_calibrationTablePath {nullptr}; 

    std::vector<std::string> m_imageList;
    std::vector<char>        m_calibrationCache;

    float* m_deviceInput {nullptr};
    bool   m_readCache {true};
    int    m_imageIndex;
};

class Int8MinMaxCalibrator : public nvinfer1::IInt8MinMaxCalibrator
{
public:
    Int8MinMaxCalibrator(
        const int &batchSize, 
        const std::string &calibrationDataPath,
        const std::string &calibrationTablePath,
        const int& inputSize,
        const int& inputH, const int& inputW);
    
    ~Int8MinMaxCalibrator() {};
    
public:
    int         getBatchSize() const noexcept override {return m_batchSize;}
    bool        getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(std::size_t &length) noexcept override;
    void        writeCalibrationCache (const void* ptr, std::size_t legth) noexcept override;

private:
    const int m_batchSize;
    const int m_inputH;
    const int m_inputW;
    const int m_inputSize;
    const int m_inputCount;
    const std::string m_calibrationTablePath {nullptr}; 

    std::vector<std::string> m_imageList;
    std::vector<char>        m_calibrationCache;

    float* m_deviceInput {nullptr};
    bool   m_readCache {true};
    int    m_imageIndex;
};

}; // namespace model
# endif __TRT_CALIBRATOR_HPP__