# ifndef __WORKER_HPP__
# define __WORKER_HPP__

# include <vector>
# include <iostream>
# include "trt_model.hpp"
# include "trt_logger.hpp"
# include "trt_classifier.hpp"


namespace worker{

class Worker
{
public:
    Worker(std::string onnxPath, logger::Level level, model::Params params);
    void inference(std::string imagePath);

public:
    std::shared_ptr<logger::Logger>                 m_logger;
    std::shared_ptr<model::Params>                  m_params;
    std::shared_ptr<model::classifier::Classifier>  m_classifier;
    std::vector<float>                              m_scores;
};

// 工厂函数, 创建并返回一个Worker对象的智能指针(传入参数与构造函数相同)
std::shared_ptr<Worker> create_worker(std::string onnxPath, logger::Level level, model::Params params);
}; // namespace worker
# endif