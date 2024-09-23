# include "model.hpp"

using namespace std;


class Logger : public nvinfer1::ILogger{
public:
    virtual void log (Severity severity, const char* msg) noexcept override{
        std::string str;
        switch (severity){
            case Severity::kINTERNAL_ERROR: str = RED    "[fatal]" CLEAR;
            case Severity::kERROR:          str = RED    "[error]" CLEAR;
            case Severity::kWARNING:        str = BLUE   "[warn]"  CLEAR;
            case Severity::kINFO:           str = YELLOW "[info]"  CLEAR;
            case Severity::kVERBOSE:        str = PURPLE "[verb]"  CLEAR;
        }
        if (severity <= Severity::kINFO)
            cout << str << string(msg) << endl;
    }
};

struct InferDeleter
{
    template <typename T>
    void operator() (T* obj) const {
        delete obj;
    }
};

template <typename T>
using make_trtunique = std::unique_ptr<T, InferDeleter>;

Model::Model(std::string onnxPath, precision prec) {

    if (getFileType(onnxPath) == ".onnx") {
        mOnnxPath = onnxPath;
    } else {
        LOGE("ERROR: %s, wrong weight or model type selected. Program terminated", getFileType(onnxPath).c_str());
    }

    if (prec == precision::FP16) {
        mPrecision = nvinfer1::DataType::kHALF;
        mEnginePath = getEnginePath(onnxPath, "FP16");
    } else if (prec == precision::INT8) {
        mPrecision = nvinfer1::DataType::kINT8;
        mEnginePath = getEnginePath(onnxPath, "INT8");
    } else {
        mPrecision = nvinfer1::DataType::kFLOAT;
        mEnginePath = getEnginePath(onnxPath, "FP32");
    }
}

bool Model::build() 
{
    if (fileExists(mEnginePath)) {
        LOG("%s has been generated!", mEnginePath.c_str());
        return true;
    } else {
        LOG("%s not found. Building engine...", mEnginePath.c_str());
    }

    // 基本组件
    Logger logger;
    auto builder = make_trtunique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network = make_trtunique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config  = make_trtunique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser  = make_trtunique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!builder || !network) {
        LOGE("ERROR: failed to create builder or network");
        return false;
    }

    // parser 解析
    // 启用TensorRT详细日志, 捕获更多细节
    config->setFlag(nvinfer1::BuilderFlag::kDEBUG);
    config->setMaxWorkspaceSize(1<<28);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    if (!parser->parseFromFile(mOnnxPath.c_str(), 1)){
        LOGE("ERROR: failed to %s", mOnnxPath.c_str());
        return false;
    }

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        LOGE("ERROR: Failed to create profile");
        return false;
    }

    // 优化策略
    #if 1
    builder->setMaxBatchSize(1);
    auto input = network->getInput(0);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 224, 224}); // 设置最小尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 224, 224}); // 设置最优尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 224, 224}); // 设置最大尺寸
    config->addOptimizationProfile(profile);
    #endif

    // 序列化保存
    auto plan    = builder->buildSerializedNetwork(*network, *config);
    auto runtime = make_trtunique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine  = make_trtunique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

    auto f = fopen(mEnginePath.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);

    LOGV("Finished building engine");

    return true;
}

bool Model::infer(std::string imagePath) {

    if (!fileExists(mEnginePath)) {
        LOGE("ERROR: %s not found", mEnginePath.c_str());
        return false;
    }

    Timer timer;
    Logger logger;
    vector<unsigned char> modelData;
    modelData = loadFile(mEnginePath);
    
    
    /*
    ===================================================================
    ============================= 基本组件 =============================
    ===================================================================
    */
    // runtime   engine   context
    // runtime: 管理和操作推理引擎的生命周期
    auto runtime    = make_trtunique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    
    // engine: 封装了模型的推理逻辑和参数
    auto engine     = make_trtunique<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    
    // context: 设置输入, 执行推理并获取输出
    auto context    = make_trtunique<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    // 指定参数
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto input_dims  = context->getBindingDimensions(0);
    auto output_dims = context->getBindingDimensions(1);

    int input_channel   = input_dims.d[1];
    int input_height    = input_dims.d[2];
    int input_width     = input_dims.d[3];
    int num_classes     = output_dims.d[1];
    int input_size      = input_channel * input_height * input_width * sizeof(float);
    int output_size     = num_classes * sizeof(float);


    /*
    ===================================================================
    ============================== 前处理 ==============================
    ===================================================================
    */
    float* input_h  = nullptr;
    float* input_d  = nullptr;
    float* output_h = nullptr;
    float* output_d = nullptr;
    CUDA_CHECK(cudaMalloc(&input_d, input_size));
    CUDA_CHECK(cudaMalloc(&output_d, output_size));
    CUDA_CHECK(cudaMallocHost(&input_h, input_size));
    CUDA_CHECK(cudaMallocHost(&output_h, output_size));

    // 测速
    timer.start_cpu();
    cv::Mat input_img;
    input_img = cv::imread(imagePath);
    if (input_img.data == nullptr) {
        LOGE("file not founded! Program terminated");
        return false;
    } else {
        LOG("Model:      %s", getFileName(mOnnxPath).c_str());
        LOG("Precision:  %s", getPrecision(mPrecision).c_str());
        LOG("Image:      %s", getFileName(imagePath).c_str());
    }

    // 前处理
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};
    cv::resize(input_img, input_img, cv::Size(input_width, input_height));

    // host端进行处理 -> norm + BGR2RGB + HWC2CWH
    int index;
    int offset_ch0 = input_width * input_height * 0; // B
    int offset_ch1 = input_width * input_height * 1; // G
    int offset_ch2 = input_width * input_height * 2; // R
    for (int i = 0; i < input_height; ++i) {
        for (int j = 0; j < input_width; ++j) {
            index = i * input_width * input_channel + j * input_channel;
            input_h[offset_ch2++] = (input_img.data[index + 0] / 255.0f - mean[0]) / std[0];
            input_h[offset_ch1++] = (input_img.data[index + 1] / 255.0f - mean[0]) / std[0];
            input_h[offset_ch0++] = (input_img.data[index + 2] / 255.0f - mean[0]) / std[0];
        }
    }
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("preprocess(resize + norm + bgr2rgb + hwc2chw + H2D)");


    /*
    ===================================================================
    =============================== 推理 ===============================
    ===================================================================
    */
    // host2device
    cudaMemcpy(input_d, input_h, input_size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    float* bingings[] = {input_d, output_d};
    
    timer.start_cpu();
    if (!context->enqueueV2((void**)bingings, stream, nullptr)) {
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("inference(enqueuev2)");


    /*
    ===================================================================
    ============================== 后处理 ==============================
    ===================================================================
    */
    cudaMemcpy(output_h, output_d, output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    timer.start_cpu();
    ImageNetLabels labels;

    // output_h: 首地址, num_classes: 偏移量
    // - output_h: 求出相对位置
    int pos = max_element(output_h, output_h + num_classes) - output_h;
    float confidence = output_h[pos] * 100;
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("postprocess(D2H + get label)");
    LOG("Inference result: %s, Confidence is %.3f%%\n", labels.imagenet_labelstring(pos).c_str(), confidence);   

    // free
    cudaFree(input_d);
    cudaFree(output_d);
    cudaStreamDestroy(stream);


    return true;
}