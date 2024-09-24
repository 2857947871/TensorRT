# include "trt_model.hpp"
# include "trt_classifier.hpp"
# include "imagenet_labels.hpp"
# include "NvInfer.h"
# include "NvOnnxParser.h"




namespace model
{
namespace classifier 
{
// classifier类任务的初始化相关内容
// input, output bindings, 分配host/device memory
void Classifier::setup(const void* data, const size_t size) {

    // 创建基本组件
    // runtime   engine   context
    m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_logger), model::destory_trt_ptr<nvinfer1::IRuntime>);
    m_engine  = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(data, size), model::destory_trt_ptr<nvinfer1::ICudaEngine>);
    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext(), model::destory_trt_ptr<nvinfer1::IExecutionContext>);
    m_inputDims   = m_context->getBindingDimensions(0);
    m_outputDims  = m_context->getBindingDimensions(1);
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    m_inputSize     = m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
    m_outputSize    = m_params->num_cls * sizeof(float);
    m_imgArea       = m_params->img.h * m_params->img.w;

    // malloc
    // 0: device   1: host
    CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], m_inputSize));
    CUDA_CHECK(cudaMallocHost(&m_outputMemory[0], m_outputSize));
    CUDA_CHECK(cudaMalloc(&m_inputMemory[1], m_inputSize));
    CUDA_CHECK(cudaMalloc(&m_outputMemory[1], m_outputSize));

    // 创建bindings
    m_bindings[0] = m_inputMemory[1];
    m_bindings[1] = m_outputMemory[1];
}

bool Classifier::preprocess_cpu() {
    /*Preprocess -- 获取mean, std*/
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    /*Preprocess -- 读取数据*/
    cv::Mat input_image;
    input_image = cv::imread(m_imagePath);
    if (input_image.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated"); 
        return false;
    }

    /*Preprocess -- 测速*/
    m_timer->start_cpu();

    /*Preprocess -- resize(默认是bilinear interpolation)*/
    cv::resize(input_image, input_image, 
               cv::Size(m_params->img.w, m_params->img.h), 0, 0, cv::INTER_LINEAR);

    /*Preprocess -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
    int index;
    int offset_ch0 = m_imgArea * 0;
    int offset_ch1 = m_imgArea * 1;
    int offset_ch2 = m_imgArea * 2;
    for (int i = 0; i < m_inputDims.d[2]; i++) {
        for (int j = 0; j < m_inputDims.d[3]; j++) {
            index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
            m_inputMemory[0][offset_ch2++] = (input_image.data[index + 0] / 255.0f - mean[0]) / std[0];
            m_inputMemory[0][offset_ch1++] = (input_image.data[index + 1] / 255.0f - mean[1]) / std[1];
            m_inputMemory[0][offset_ch0++] = (input_image.data[index + 2] / 255.0f - mean[2]) / std[2];
        }
    }

    /*Preprocess -- 将host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("preprocess(CPU)");

    return true;
}

bool Classifier::preprocess_gpu() {
    /*Preprocess -- 获取mean, std*/
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    /*Preprocess -- 读取数据*/
    cv::Mat input_image;
    input_image = cv::imread(m_imagePath);
    if (input_image.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated"); return false;
    }

    /*Preprocess -- 测速*/
    m_timer->start_gpu();
    
    /*Preprocess -- 使用GPU进行双线性插值, 并将结果返回到m_inputMemory中*/
    process::preprocess_resize_gpu(input_image, m_inputMemory[1],
                                   m_params->img.h, m_params->img.w, 
                                   mean, std, process::tactics::GPU_BILINEAR);

    m_timer->stop_gpu();
    m_timer->duration_gpu("preprocess(GPU)");
    return true;
}


bool Classifier::postprocess_cpu() {
    
    /*Postprocess -- 测速*/
    m_timer->start_cpu();

    /*Postprocess -- 将device上的数据移动到host上*/
    int output_size    = m_params->num_cls * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    /*Postprocess -- 寻找label*/
    ImageNetLabels labels;
    int pos = max_element(m_outputMemory[0], m_outputMemory[0] + m_params->num_cls) - m_outputMemory[0];
    float confidence = m_outputMemory[0][pos] * 100;

    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("postprocess(CPU)");

    LOG("Inference result: %s", labels.imagenet_labelstring(pos).c_str());   
    LOG("Confidence is %.3f%%\n", confidence);   
    return true;
}

bool Classifier::postprocess_gpu() {
    /*
        由于classification task的postprocess比较简单，所以CPU/GPU的处理这里用一样的
        对于像yolo这种detection model, postprocess会包含decode, nms这些处理。可以选择在CPU还是在GPU上跑
    */
    return postprocess_cpu();

}

std::shared_ptr<Classifier> make_classifier(
    std::string onnx_path, logger::Level level, model::Params params)
{
    auto classifier = make_shared<Classifier>(onnx_path, level, params);
    classifier->init_model();
    return classifier;
}

}; // namespace classifier
}; // namespace model
