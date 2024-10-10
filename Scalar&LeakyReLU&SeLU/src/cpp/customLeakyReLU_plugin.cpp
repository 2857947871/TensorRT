# include <map>
# include <cstring>
# include "customLeakyReLU_plugin.hpp"
using namespace nvinfer1;

// customScalar的核函数接口部分 
void customLeakyReLUImply(const float* x, float* y, const float slope, int element_nums);

// 命名空间custom(与python中的custom::customScalar一致)
namespace custom
{
// ================================================================
// ======================= 注册PluginCreator =======================
// ================================================================
REGISTER_TENSORRT_PLUGIN(CustomLeakyReLUPluginCreator);

// ================================================================
// ========================= 静态变量的申明 ========================= 
// ================================================================
PluginFieldCollection   CustomLeakyReLUPluginCreator::mFC {};
std::vector<PluginField> CustomLeakyReLUPluginCreator::mAttrs;

// ================================================================
// =================== CustomLeakyReLUPlugin实现部分 ===================
// ================================================================
// parse 时候用的构造函数
CustomLeakyReLUPlugin::CustomLeakyReLUPlugin(const std::string &name, float slope):
    mName(name)
{
    mParams.slope = slope;
}

// clone&deseriaze 时候用的构造函数
CustomLeakyReLUPlugin::CustomLeakyReLUPlugin(const std::string &name, const void* buffer, size_t length):
    mName(name)
{
    memcpy(&mParams, buffer, sizeof(mParams));
}
CustomLeakyReLUPlugin::~CustomLeakyReLUPlugin()
{
    /* 这里的析构函数不需要做任何事情, 生命周期结束的时候会自动调用terminate和destroy */
    return;
}

// 获取名字与版本
const char* CustomLeakyReLUPlugin::getPluginType() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return PLUGIN_NAME;
}
const char* CustomLeakyReLUPlugin::getPluginVersion() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return PLUGIN_VERSION;
}

// 获取plugin的输出
int32_t CustomLeakyReLUPlugin::getNbOutputs() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return 1;
}
size_t CustomLeakyReLUPlugin::getSerializationSize() const noexcept
{
    /* 如果把所有的参数给放在mParams中的话, 一般来说所有插件的实现差不多一致 */
    return sizeof(mParams);
}
const char* CustomLeakyReLUPlugin::getPluginNamespace() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return mNamespace.c_str();
}
DataType CustomLeakyReLUPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return inputTypes[0];
}
DimsExprs CustomLeakyReLUPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return inputs[0];
}
size_t CustomLeakyReLUPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    /* 一般来说会使用builder创建时用的workspaceSize所以这里一般什么都不做 */
    return 0;
}
int32_t CustomLeakyReLUPlugin::initialize() noexcept
{
    /* 这个一般会根据情况而定, 建议每个插件都有一个自己的实现 */
    return 0;
}
void CustomLeakyReLUPlugin::terminate() noexcept 
{
    /* 
     * 这个是析构函数调用的函数。一般和initialize配对的使用
     * initialize分配多少内存, 这里就释放多少内存
    */
    return;
}
void CustomLeakyReLUPlugin::serialize(void *buffer) const noexcept
{
    /* 序列化也根据情况而定, 每个插件自己定制 */
    memcpy(buffer, &mParams, sizeof(mParams));
    return;

}

void CustomLeakyReLUPlugin::destroy() noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    delete this;
    return;
}

// 核心, plugin调用kernel
int32_t CustomLeakyReLUPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, 
    const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    /*
     * Plugin的核心的地方。每个插件都有一个自己的定制方案
     * Plugin直接调用kernel的地方
    */
    int nElements = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++){
        nElements *= inputDesc[0].dims.d[i];
    }

    customLeakyReLUImply(
            static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]), 
            mParams.slope,
            nElements);

    return 0;
}

IPluginV2DynamicExt* CustomLeakyReLUPlugin::clone() const noexcept
{
    /* 克隆一个Plugin对象, 所有的插件的实现都差不多*/
    auto p = new CustomLeakyReLUPlugin(mName, &mParams, sizeof(mParams));
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}
bool CustomLeakyReLUPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    /* 
     * 设置这个Plugin支持的Datatype以及TensorFormat, 每个插件都有自己的定制
     * 作为案例展示, 这个customScalar插件只支持FP32, 如果需要扩展到FP16以及INT8, 需要在这里设置
    */
    
    switch (pos) {
    case 0:
        return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[1].type == DataType::kFLOAT && inOut[1].format == TensorFormat::kLINEAR;
    default:
        return false;
    }
    return false;
}

void CustomLeakyReLUPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    /* 一般不需要做任何使用, 所有插件实现都差不多 */
    return;
}
void CustomLeakyReLUPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    /* 所有插件的实现都差不多 */
    mNamespace = pluginNamespace;
    return;
}
void CustomLeakyReLUPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept 
{
    /* 一般不需要做任何使用, 所有插件实现都差不多 */
    return;
}
void CustomLeakyReLUPlugin::detachFromContext() noexcept 
{
    /* 一般不需要做任何使用, 所有插件实现都差不多 */
    return;
}

// ================================================================
// =================== CustomLeakyReLUPluginCreator ==================
// ================================================================
CustomLeakyReLUPluginCreator::CustomLeakyReLUPluginCreator()
{
    /* 
     * 每个插件的Creator构造函数需要定制, 主要就是获取参数以及传递参数
     * 初始化creator中的PluginField以及PluginFieldCollection
     * - PluginField:            负责获取onnx中的参数
     * - PluginFieldCollection：  负责将onnx中的参数传递给Plugin
    */

    // static std::vector<PluginField> mAttrs;        //用来保存这个插件op所需要的权重和参数, 从onnx中获取, 同样在parse的时候使用
    // static PluginFieldCollection    mFC;           //接受plugionFields传进来的权重和参数, 并将信息传递给Plugin, 内部通过createPlugin来创建带参数的plugin

    // mAttrs 是一个 std::vector<PluginField>, 用于存储插件所需的字段信息。
    
    // PluginField("slope", nullptr, PluginFieldType::kFLOAT32, 1):
    //  "slope" 是字段的名称。
    //  nullptr 表示字段的初始值为空。
    //  PluginFieldType::kFLOAT32 指定字段的数据类型为 float32。
    //  1 表示字段的长度为1（即这是一个单一的标量值）。
    mAttrs.emplace_back(PluginField("slope", nullptr, PluginFieldType::kFLOAT32, 1));
    
    // mFC 是一个 PluginFieldCollection 类型的对象, 用来描述插件所需的所有字段集合
    // 初始化 mFC, 使其包含 mAttrs 向量中的所有字段信息。
    mFC.nbFields = mAttrs.size();
    mFC.fields   = mAttrs.data();
}

CustomLeakyReLUPluginCreator::~CustomLeakyReLUPluginCreator()
{
    /* 一般不需要做任何使用, 所有插件实现都差不多 */
}
const char* CustomLeakyReLUPluginCreator::getPluginName() const noexcept
{
    /* 所有插件实现都差不多 */
    return PLUGIN_NAME;
}
const char* CustomLeakyReLUPluginCreator::getPluginVersion() const noexcept 
{
    /* 所有插件实现都差不多 */
    return PLUGIN_VERSION;
}
const char* CustomLeakyReLUPluginCreator::getPluginNamespace() const noexcept
{
    /* 所有插件实现都差不多 */
    return mNamespace.c_str();
}

IPluginV2* CustomLeakyReLUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept 
{
    /*
     * 通过Creator创建一个Plugin的实现, 这个时候会通过mFC中取出需要的参数, 并实例化一个Plugin
     * 这个案例中, 参数有slope。从fc中取出来对应的数据来初始化这个plugin
    */
    float slope = 0;
    std::map<std::string, float*> paramMap = {{"slope", &slope}};

    for (int i = 0; i < fc->nbFields; i++) {
        if (paramMap.find(fc->fields[i].name) != paramMap.end()){
            *paramMap[fc->fields[i].name] = *reinterpret_cast<const float*>(fc->fields[i].data);
        }
    }
    return new CustomLeakyReLUPlugin(name, slope);
}

IPluginV2* CustomLeakyReLUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    /* 反序列化插件其实就是实例化一个插件, 所有插件实现都差不多 */
    return new CustomLeakyReLUPlugin(name, serialData, serialLength);
}
void CustomLeakyReLUPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    /* 所有插件实现都差不多 */
    mNamespace = pluginNamespace;
    return;
}
const PluginFieldCollection* CustomLeakyReLUPluginCreator::getFieldNames() noexcept
{
    /* 所有插件实现都差不多 */
    return &mFC;
}

} // namespace custom

