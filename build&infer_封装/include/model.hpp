# ifndef __MODEL_HPP__
# define __MODEL_HPP__

# include <memory>
# include <string>
# include <cassert>
# include <iostream>
# include <type_traits>
# include "NvInfer.h"
# include "NvOnnxParser.h"
# include "cuda_runtime.h"
# include "opencv2/core/core.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/opencv.hpp"
# include "utils.hpp"
# include "model.hpp"
# include "timer.hpp"
# include "imagenet_labels.hpp"


// 封装
class Model
{
public:
    enum precision {
        FP32,
        FP16,
        INT8
    };

public:
    Model(std::string onnxPath, precision prec);
    bool build();
    bool infer(std::string imgPath);
    void print_network(nvinfer1::INetworkDefinition &network, bool optimized);

private:
    // bool preprocess();
    // bool build_from_onnx();

private:
    std::string mOnnxPath = "";
    std::string mEnginePath = "";
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    float* mInputHost;
    float* mInputDevice;
    float* mOutputHost;
    float* mOutputDevice;
    int mInputSize;
    int mOutputSize;
    nvinfer1::DataType mPrecision;
};
# endif