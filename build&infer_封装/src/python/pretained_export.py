import torch
import torchvision
import onnx
import onnxsim
import argparse

# 构建模型
class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.softmax = torch.nn.Softmax()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.softmax(x)
        return x

# pth -> onnx
def export_onnx(model, file, input, dynamic_axes=None):
    model.cuda()
    torch.onnx.export(
        model           = model,
        args            = (input,),
        f               = file,
        input_names     = ["input0"],
        output_names    = ["output"],
        dynamic_axes    = dynamic_axes,
        opset_version   = 11
    )
    print("finish onnx export")

    # check
    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    # use onnx-simplifier to simplify the onnx
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)

def get_backbone(type, dir):
    # Here, we only exported pretrained model in torchvision model zoo
    if type == "resnet":
        backbone = torchvision.models.resnet50(pretrained=True)
        file  = dir + "resnet50.onnx"
    elif type == "vgg":
        backbone = torchvision.models.vgg19_bn(pretrained=True)
        file  = dir + "vgg19.onnx"
    elif type == "mobilenet":
        backbone = torchvision.models.mobilenet_v2(pretrained=True)
        file  = dir + "mobilenetV2.onnx"
    elif type == "squeezenet":
        backbone = torchvision.models.squeezenet1_0(pretrained=True)
        file  = dir + "squeezenet1_0.onnx"
    elif type == "densenet":
        backbone = torchvision.models.densenet161(pretrained=True)
        file  = dir + "densenet161.onnx" # Too long to see
    elif type == "shufflenet":
        backbone = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        file  = dir + "shufflenetV2.onnx"
    return backbone, file

def main(args):
    type        = args.type
    dir         = args.dir
    input       = torch.rand(1, 3, 224, 224, device='cuda')
    backbone, file = get_backbone(type, dir)
    model = Model(backbone=backbone)

    export_onnx(model, file, input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, default="resnet")
    parser.add_argument("-d", "--dir", type=str, default="./model/onnx/")
    
    opt = parser.parse_args()
    main(opt)