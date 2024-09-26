# include <memory>
# include "trt_model.hpp"
# include "trt_worker.hpp"
# include "trt_logger.hpp"
# include "trt_classifier.hpp"

namespace worker
{
    Worker::Worker(std::string onnxPath, logger::Level level, model::Params params) {

        m_logger = logger::create_logger(level);

        // 根据不同task选择创建不同的trt_model
        // TODO: detection, segmentation
        if (params.task == model::task_type::CLASSIFICATION)
        {
            m_classifier = model::classifier::make_classifier(onnxPath, level, params);
        }
    }

    void Worker::inference(std::string imagePath) {

        if (m_classifier != nullptr) {
            m_classifier->load_image(imagePath);
            m_classifier->inference();
        }
    }

std::shared_ptr<Worker> create_worker(std::string onnxPath, logger::Level level, model::Params params)
{
    // 使用智能指针来创建一个实例
    return std::make_shared<Worker>(onnxPath, level, params);
}
}; // namespace worker