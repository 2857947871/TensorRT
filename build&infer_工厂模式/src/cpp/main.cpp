# include <string>
# include <iostream>
# include "trt_model.hpp"
# include "trt_worker.hpp"
# include "trt_logger.hpp"


int main() 
{
    // 工厂模式: 调用即初始化
    std::string onnxPath = "models/onnx/resnet50.onnx";
    
    auto level          = logger::Level::VERB;
    auto params         = model::Params();

    params.img          = {224, 224, 3};
    params.num_cls      = 1000;
    params.task         = model::task_type::CLASSIFICATION;
    params.dev          = model::device::GPU;
    params.tac          = process::tactics::GPU_BILINEAR;
    params.prec         = model::precision::INT8;
    params.cal          = model::calibrator::Entropy;

    // 创建worker实例, 在创建的时候完成初始化
    auto worker = worker::create_worker(onnxPath, level, params);

    // 推理
    # if 1
    worker->inference("data/cat.png");
    worker->inference("data/fox.png");
    worker->inference("data/wolf.png");
    worker->inference("data/eagle.png");
    worker->inference("data/gazelle.png");
    worker->inference("data/tiny-cat.png");
    # endif

    return 0;
}
