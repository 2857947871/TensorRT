# ifndef __TRT_CLASSIFIER_HPP__
# define __TRT_CLASSIFIER_HPP__

# include <iostream>
# include "NvInfer.h"
# include "trt_model.hpp"
# include "trt_logger.hpp"


namespace model
{
namespace classifier
{
class Classifier : public Model
{
public:
    // 直接调用父类的构造函数 
    Classifier(std::string onnx_path, logger::Level level, Params params) : 
        Model(onnx_path, level, params) {};

public:
    // 每种任务都要重写虚函数(前后处理以及初始化)
    virtual void setup(const void* data, const std::size_t size) override;
    virtual bool preprocess_cpu() override;
    virtual bool preprocess_gpu() override;
    virtual bool postprocess_cpu() override;
    virtual bool postprocess_gpu() override;

private:
    float m_confidence;
    std::string m_label;
    int m_imgArea;
    int m_inputSize;
    int m_outputSize;
};

// 工厂函数, 创建并返回一个Classifier对象的智能指针(传入参数与构造函数相同)
std::shared_ptr<Classifier> make_classifier(
    std::string onnx_path, logger::Level level, Params params);
} // namespace classifier
} // namespace model
# endif