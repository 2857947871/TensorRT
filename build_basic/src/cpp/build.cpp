/*
==================================================
========= basic版本的build, 无量化以及封装 ==========
==================================================
*/
# include <memory>
# include <cassert>
# include <buffers.h>
# include <iostream>
# include "NvInfer.h"
# include "NvOnnxParser.h"


// 保存 enging
void saveEngine(const std::string filename, std::shared_ptr<nvinfer1::IHostMemory> engine)
{
    std::ofstream outfile(filename, std::ios::binary);
    assert(outfile.is_open() && "save file failed!");
    outfile.write((char *)engine->data(), engine->size());
    outfile.close();
};

int main()
{
    /*
    ==================================================
    ================== 1. 创建基本组件 =================
    ==================================================
    */
    // builder network config parser
    auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());  
    auto parser = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!builder || !network) {
        std::cerr << "Failed to create builder or network" << std::endl;
        return -1;
    }

    /*
    ==================================================
    ============== 2. builder -> network =============
    ==================================================
    */
    // 使用解析器 -> network
    char* onnx_file_path = {"./model/onnx/resnet50.onnx"};
    auto parsed = parser->parseFromFile(onnx_file_path, static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        std::cout << "Failed to parse onnx file" << std::endl;
        return -1;
    }

    /*
    ==================================================
    ============== 3. builder -> config ==============
    ==================================================
    */
    // 创建 profile 与 profileStream(设置 profile)
    auto profile = builder->createOptimizationProfile();
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profile || !profileStream) {
        std::cerr << "Failed to create profile or profileStream" << std::endl;
        return -1;
    }

    // 设置维度
    // 如果是onnx为静态维度 -> 最小最大最优也为固定值
    auto input = network->getInput(0);
    builder->setMaxBatchSize(1);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 224, 224}); // 设置最小尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 224, 224}); // 设置最优尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 224, 224}); // 设置最大尺寸
    config->addOptimizationProfile(profile);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 32);
    
    // profile -> config
    config->setProfileStream(*profileStream);

    /*
    ==================================================
    ============== 4. builder -> engine ==============
    ==================================================
    */
    auto engine = std::shared_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!engine) {
        std::cerr << "Failed to create engine" << std::endl;
        return -1;
    }

    // ========================== 5. 序列化保存engine ==========================
    saveEngine("./model/engine/resnet50.engine", engine);

    // ========================== 6. free ==========================
    std::cerr << "Engine build success!" << std::endl;

    return 0;
}