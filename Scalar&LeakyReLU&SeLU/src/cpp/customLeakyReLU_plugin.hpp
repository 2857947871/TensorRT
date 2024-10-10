/*
==========================================
==继承IPluginV2DynamicExt和IPluginCreator==
==========================================
*/

#ifndef __CUSTOM_LEAKYRELU_PLUGIN_HPP
#define __CUSTOM_LEAKYRELU_PLUGIN_HPP

#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include <NvInfer.h>
#include <string>
#include <vector>

using namespace nvinfer1;

/*
    # symbolic(g, input, args)
    @staticmethod
    def symbolic(g, x, slope):
        return g.op(
                    # 算子名称
                    "custom::customLeakyReLU",

                    # 输入参数
                    x,
                    slope_f=slope,
                    
                    # 属性
                    param1_s="my_leakyReLU"
                    )
*/

// 定义命名空间(plugin在此命名空间下, 与python中的custom::customLeakyReLU一致)
namespace custom{

// PLUGIN_NAME(插件名称, 与python中的custom::customLeakyReLU一致)
static const char* PLUGIN_NAME = "customLeakyReLU";

// PLUGIN_VERSION(插件版本)
static const char* PLUGIN_VERSION = "1";

// 创建两个类, Plugin类, PluginCreator类
//  Plugin类: 继承IPluginV2DynamicExt, plugin的具体实现
//  PluginCreator类: 继承IPluginCreator, plugin的创建

// Plugin类, plugin的具体实现
class CustomLeakyReLUPlugin : public IPluginV2DynamicExt {
public:
    /*
      编译的过程中有三次创建插件实例的过程
      1. parse: 第一次读取 .onnx 来 parse 这个插件, 读取参数并转换为TRT格式
      2. clone: parse 后, TRT会为了优化这个 plugin -> clone 很多副本
      3. deseriaze: 序列化好的 plugin 进行反序列化的时候也需要创建插件的实例
    */

    // 默认构造函数
    CustomLeakyReLUPlugin() = delete;  // 禁止默认构造函数

    // parse 时候用的构造函数
    CustomLeakyReLUPlugin(const std::string &name, float slope); 

    // clone&deseriaze 时候用的构造函数
    CustomLeakyReLUPlugin(const std::string &name, const void* buffer, size_t length);

    ~CustomLeakyReLUPlugin();

public:
    /* 有关获取plugin信息的方法 */
    // 继承自IPluginV2DynamicExt, 大部分都需要重写IPluginV2DynamicExt的方法
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    size_t      getSerializationSize() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    DataType    getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    DimsExprs   getOutputDimensions(int32_t outputIndex, const DimsExprs* input, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    size_t      getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;

    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    void        serialize(void *buffer) const noexcept override;
    void        destroy() noexcept override;
    int32_t     enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* ionputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override; // 实际插件op执行的地方, 具体实现forward的推理的CUDA/C++实现会放在这里面
    IPluginV2DynamicExt* clone() const noexcept override;

    bool        supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOuts, int32_t nbInputs, int32_t nbOutputs) noexcept override; //查看pos位置的索引是否支持指定的DataType以及TensorFormat
    void        configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override; //配置插件, 一般什么都不干
    void        setPluginNamespace(const char* pluginNamespace) noexcept override;

    void        attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void        detachFromContext() noexcept override;

private:
    const std::string mName;
    std::string       mNamespace;
    
    // plugin 所需要的参数
    struct  {
        float slope;
    } mParams;
};


// PluginCreator类, plugin的创建
class CustomLeakyReLUPluginCreator : public IPluginCreator {
public:
    CustomLeakyReLUPluginCreator();  //初始化mFC以及mAttrs
    ~CustomLeakyReLUPluginCreator();

    const char*                     getPluginName() const noexcept override;
    const char*                     getPluginVersion() const noexcept override;
    const PluginFieldCollection*    getFieldNames() noexcept override;
    const char*                     getPluginNamespace() const noexcept override;
    IPluginV2*                      createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;  //通过包含参数的mFC来创建Plugin。调用上面的Plugin的构造函数
    IPluginV2*                      deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void                            setPluginNamespace(const char* pluginNamespace) noexcept override;
      
private:
    static PluginFieldCollection    mFC;           //接受plugionFields传进来的权重和参数, 并将信息传递给Plugin, 内部通过createPlugin来创建带参数的plugin
    static std::vector<PluginField> mAttrs;        //用来保存这个插件op所需要的权重和参数, 从onnx中获取, 同样在parse的时候使用
    std::string                     mNamespace;
    
};
} // namespace custom

#endif __CUSTOM_LEAKYRELU_PLUGIN_HPP

