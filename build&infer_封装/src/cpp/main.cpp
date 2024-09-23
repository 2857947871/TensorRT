# include <iostream>
# include "model.hpp"
# include "utils.hpp"

int main() {

    // 命名格式: 直接以文件夹开头, 不要加 ./
    Model model("models/onnx/resnet50.onnx", Model::precision::FP32);

    if(!model.build()){
        LOGE("fail in building model");
        return 0;
    }
#if 1
    if(!model.infer("data/fox.png")){
        LOGE("fail in infering model");
        return 0;
    }
    if(!model.infer("data/cat.png")){
        LOGE("fail in infering model");
        return 0;
    }

    if(!model.infer("data/eagle.png")){
        LOGE("fail in infering model");
        return 0;
    }

    if(!model.infer("data/gazelle.png")){
        LOGE("fail in infering model");
        return 0;
    }
#endif
    return 0;
}