import torch
import torch.onnx as onnx

input_path = './log/train_1/epoch15_avgtrainloss0.0148436_avgvalloss0.0162469.pth'
output_path = './onnx/epoch15_avgtrainloss0.0148436_avgvalloss0.0162469.onnx'


model = torch.load(input_path)
model.eval().cuda()

img = torch.randn(1,3,512,512).cuda()
input_names = ["input"]  # 名字随意
output_names = ["output"] # 名字随意

onnx.export(
    model, 
    img, 
    output_path, 
    verbose=True, 
    input_names=input_names, 
    output_names=output_names, 
    opset_version=11
)