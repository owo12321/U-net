import torch
import cv2
import numpy as np

model_file = './log/train_1/epoch15_avgtrainloss0.0148436_avgvalloss0.0162469.pth'
in_file = './data/ciwaquexian/MT_Blowhole/train/exp1_num_4727.jpg'
out_file = './output/test.png'

raw = cv2.imread(in_file)
array = cv2.resize(raw, (512, 512))                 # 调整成(512,512,3)
array = np.swapaxes(np.swapaxes(array, 0, 2), 1, 2) # 调整成(3,512,512)
array = array[np.newaxis, :, :, :]                  # 调整成(1,3,512,512)

# 转成输入模型的tensor
tensor = torch.from_numpy(array).to(device='cuda:0', dtype=torch.float32)
print(raw.shape, array.shape, tensor.size())

# 加载模型
model = torch.load(model_file)
model.eval()

# 预测
with torch.no_grad():
    output = model(tensor)
    print(output.size())

# 结果转成numpy
res = output.cpu().numpy()[0][0]
print(res.shape)

# 把结果变成0到255
flag, res = cv2.threshold(res, 0.15, 255, cv2.THRESH_BINARY)
res = res.astype(np.uint8)

# 保存图片
cv2.imwrite(out_file, res)