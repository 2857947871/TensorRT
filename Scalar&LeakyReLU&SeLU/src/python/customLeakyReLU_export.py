import torch
import torch.onnx
import torch.nn as nn
import onnx
import onnxsim

# check
def check(model_my, model_gt):
    
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
            [1,   0,   1],
            [1,   1,   -1]
        ]
    ], dtype=torch.float32).view(2, 1, 3, 3)

    output_my = model_my(input)
    output_gt = model_gt(input)
    
    print(f"inference output = \n{output_my}")
    print(f"inference output = \n{output_gt}")
    if (output_my - output_gt).abs().max() < 1e-5:
        print("model_my is correct!")
    else:
        print("model_my is wrong!")

"""
=====================================================================================================
============================================ 1. 自定义算子 ============================================
=====================================================================================================
"""
# 省略C++实现部分，这里只给出Pytorch实现
class LeakyReLUImply(torch.autograd.Function):

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
    # 参数与symbolic保持一致
    # 因为是自行实现的算子，所以return的结果也需要自行实现,
    # 如果是映射, 直接return pytorch的算子即可
    @staticmethod
    def forward(ctx, x, slope):
        return torch.where(x > 0, x, x * slope)
    
# 封装成torch.nn.Module
# 也可以直接使用, 但是要使用apply调用而不是直接实例化
class LeakyReLU(nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.slope = slope
    
    def forward(self, x):
        return LeakyReLUImply.apply(x, self.slope)

# 构造模型
class Model_my(nn.Module):
    def __init__(self):
        super().__init__()
        self.leakyrelu = LeakyReLU(0.01)
    
    def forward(self, x):
        x = self.leakyrelu(x)
        return x

class Model_gt(nn.Module):
    def __init__(self):
        super().__init__()
        self.leakyrelu = nn.LeakyReLU(0.01)
    
    def forward(self, x):
        x = self.leakyrelu(x)
        return x

# check
model_my = Model_my()
model_gt = Model_gt()
check(model_my, model_gt)


"""
=====================================================================================================
============================================ 2. 转成ONNX ============================================
=====================================================================================================
"""
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
        [1,   0,   1],
        [1,   1,   -1]
    ]
], dtype=torch.float32).view(2, 1, 3, 3)

torch.onnx.export(
    # 待转换模型
    model_my,
    
    # 输入
    (input,),

    # 保存路径
    "./models/onnx/sample_customLeakyReLU.onnx",

    # 输入输出
    input_names=["input"],
    output_names=["output"],

    opset_version=11
)

# check
model_onnx = onnx.load("./models/onnx/sample_customLeakyReLU.onnx")
onnx.checker.check_model(model_onnx)

# simplify
print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, "assert check failed"
onnx.save(model_onnx, "./models/onnx/sample_customLeakyReLU.onnx")