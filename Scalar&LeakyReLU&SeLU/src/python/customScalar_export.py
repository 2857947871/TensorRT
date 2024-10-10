import torch
import torch.onnx
import torch.nn as nn
import onnx
import onnxsim



"""
=====================================================================================================
============================================ 1. 自定义算子 ============================================
=====================================================================================================
"""
# 省略C++实现部分，这里只给出Pytorch实现
class CustomScalarImply(torch.autograd.Function):

    # symbolic(g, input, args)
    @staticmethod
    def symbolic(g, x, r, s):
        return g.op(
                    # 算子名称 
                    "custom::customScalar",
                    
                    # 输入参数
                    x,
                    scalar_f=r,
                    scale_f=s,

                    # 属性
                    param1_s="my_custom_scalar",
                    )
    # forward(ctx, x, r, s)
    # 参数与symbolic保持一致
    # 因为是自行实现的算子，所以return的结果也需要自行实现, 
    # 如果是映射, 直接return pytorch的算子即可
    @staticmethod
    def forward(ctx, x, r, s):
        return (x + r) * s

# 封装成torch.nn.Module
# 也可以直接使用, 但是要使用apply调用而不是直接实例化
class CustomScalar(nn.Module):
    def __init__(self, r, s):
        super().__init__()
        self.scalar = r
        self.scale  = s

    def forward(self, x):
        return CustomScalarImply.apply(x, self.scalar, self.scale)

# 构造模型
class Model_my(nn.Module):
    def __init__(self):
        super().__init__()

        # self.conv = nn.Conv2d(1, 3, 3, 3, padding=1)
        # self.conv.weight.data.fill_(1)
        # self.conv.bias.data.fill_(0)
        self.custom = CustomScalar(1, 10)

    def forward(self, x):
        # x = self.conv(x)
        x = self.custom(x)
        return x

# 验证自定义算子
model = Model_my()
model.eval()
input = torch.tensor([
    # batch 0
    [
        [1,   1,   1],
        [1,   1,   1],
        [1,   1,   1],
    ],
    # batch 1
    [
        [-1,   1,   1],
        [1,    0,   1],
        [1,    1,  -1]
    ]
], dtype=torch.float32).view(2, 1, 3, 3)

# y = (x + 1) * 10
output = model(input)
print(output)

"""
=====================================================================================================
============================================ 2. 转成ONNX ============================================
=====================================================================================================
"""
input = torch.tensor([[[
    [0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
    [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
    [0.6387, 0.5247, 0.6826, 0.3051, 0.4635],
    [0.4550, 0.5725, 0.4980, 0.9371, 0.6556],
    [0.3138, 0.1980, 0.4162, 0.2843, 0.3398]]]])
torch.onnx.export(
    
    # 待转换模型
    model,

    # 输入数据
    (input,),

    # 保存路径
    "./models/onnx/sample_customScalar.onnx",

    # 输入&输出
    input_names=["input"],
    output_names=["output"],

    opset_version=11
)

# check the exported onnx model
model_onnx = onnx.load("./models/onnx/sample_customScalar.onnx")
onnx.checker.check_model(model_onnx)

# use onnx-simplifier to simplify the onnx
print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, "assert check failed"
onnx.save(model_onnx, "./models/onnx/sample_customScalar.onnx")