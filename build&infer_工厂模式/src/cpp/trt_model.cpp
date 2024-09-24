# include <string>
# include <vector>
# include <memory>
# include <iostream>
# include "trt_timer.hpp"
# include "trt_model.hpp"
# include "trt_logger.hpp"
# include "trt_classifier.hpp"
# include "NvInfer.h"
# include "NvOnnxParser.h"


namespace model
{
Model::Model(std::string onnx_path, logger::Level level, Params params) {
    m_onnxPath      = onnx_path;
    m_params        = new Params(params);
    m_enginePath    = getEnginePath(onnx_path);
    m_workspaceSize = WORKSPACESIZE;
    m_logger        = std::make_shared<logger::Logger>(level);
    m_timer         = std::make_shared<timer::Timer>();
}

void Model::load_image(std::string image_path) {
    if (!fileExists(image_path)) {
        LOGE("EORROR: %f not found\n", image_path);
    } else {
        m_imagePath = image_path;
        LOG("Model: %s", getFileName(m_onnxPath).c_str());
        LOG("Image: %s", getFileName(m_imagePath).c_str());
    }
}

void Model::init_model() {

    // engine检测, 没有必要重复生成
    if (!fileExists(m_enginePath)) {
        LOGV("%s not found. Building trt engine...", m_enginePath.c_str());
        build_engine();
    } else {
        LOGV("%s has been found. loading trt engine...", m_enginePath.c_str());
        load_engine();
    }
}

bool Model::build_engine() {

    // 创建基本组件
    auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*m_logger), model::destory_trt_ptr<nvinfer1::IBuilder>);
    auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1), model::destory_trt_ptr<nvinfer1::INetworkDefinition>);
    auto config  = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig(), model::destory_trt_ptr<nvinfer1::IBuilderConfig>);
    auto parser  = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *m_logger), model::destory_trt_ptr<nvonnxparser::IParser>);

    // 设置参数
    config->setMaxWorkspaceSize(m_workspaceSize);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    // parser解析onnx
    if (!parser->parseFromFile(m_onnxPath.c_str(), 1)){
        return false;
    }

    // 优化
    if (builder->platformHasFastFp16() && m_params->prec == model::FP16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }

    #if 1
    builder->setMaxBatchSize(1);
    auto input = network->getInput(0);
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        LOGE("ERROR: Failed to create profile");
        return false;
    }
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 224, 224}); // 设置最小尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 224, 224}); // 设置最优尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 224, 224}); // 设置最大尺寸
    config->addOptimizationProfile(profile);
    # endif
    
    // 保存序列化后的engine
    auto plan = builder->buildSerializedNetwork(*network, *config);
    save_plan(*plan);

    // 根据runtime初始化engine, context, 以及memory
    // 此处的setup: TensorRT自动优化网络
    setup(plan->data(), plan->size());

    // 把优化前和优化后的各个层的信息打印出来
    LOGV("Before TensorRT optimization");
    print_network(*network, false);
    LOGV("After TensorRT optimization");
    print_network(*network, true);

    return true;
}

bool Model::load_engine() {

    // load一个engine时, 直接初始化
    //  1. 反序列化engine
    //  2. 创建context
    //  3. 创建推理所用的stream
    //  4. malloc推理所用的memory
    if (!fileExists(m_enginePath)) {
        LOGE("engine does not exits! Program terminated");
        return false;
    }

    std::vector<unsigned char> modelData;
    modelData = loadFile(m_enginePath);

    // 根据runtime初始化engine, context, 以及memory
    setup(modelData.data(), modelData.size());

    return true;
}

void Model::inference() {
    if (m_params->dev == CPU) {
        preprocess_cpu();
    } else {
        preprocess_gpu();
    }

    enqueue_bindings();

    if (m_params->dev == CPU) {
        postprocess_cpu();
    } else {
        postprocess_gpu();
    }
}

bool Model::enqueue_bindings() {
    m_timer->start_gpu();
    if (!m_context->enqueueV2((void**)m_bindings, m_stream, nullptr)){
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
    m_timer->stop_gpu();
    m_timer->duration_gpu("trt-inference(GPU)");
    return true;
}

void Model::save_plan(nvinfer1::IHostMemory& plan) {
    auto f = fopen(m_enginePath.c_str(), "wb");
    fwrite(plan.data(), 1, plan.size(), f);
    fclose(f);
}

void Model::print_network(nvinfer1::INetworkDefinition &network, bool optimized) {

    int inputCount = network.getNbInputs();
    int outputCount = network.getNbOutputs();
    std::string layer_info;

    for (int i = 0; i < inputCount; i++) {
        auto input = network.getInput(i);
        LOGV("Input info: %s:%s", input->getName(), printTensorShape(input).c_str());
    }

    for (int i = 0; i < outputCount; i++) {
        auto output = network.getOutput(i);
        LOGV("Output info: %s:%s", output->getName(), printTensorShape(output).c_str());
    }
    
    int layerCount = optimized ? m_engine->getNbLayers() : network.getNbLayers();
    LOGV("network has %d layers", layerCount);

    if (!optimized) {
        for (int i = 0; i < layerCount; i++) {
            char layer_info[1000];
            auto layer   = network.getLayer(i);
            auto input   = layer->getInput(0);
            int n = 0;
            if (input == nullptr){
                continue;
            }
            auto output  = layer->getOutput(0);

            LOGV("layer_info: %-40s:%-25s->%-25s[%s]", 
                layer->getName(),
                printTensorShape(input).c_str(),
                printTensorShape(output).c_str(),
                getPrecision(layer->getPrecision()).c_str());
        }

    } else {
        auto inspector = std::shared_ptr<nvinfer1::IEngineInspector>(m_engine->createEngineInspector());
        for (int i = 0; i < layerCount; i++) {
            LOGV("layer_info: %s", inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
        }
    }
}

}; // namespace model