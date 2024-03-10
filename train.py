import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import os

from model import Unet
from ciwablowholedataset import CiwaBlowholeDataset


# device = 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# 创建模型
model = Unet()
model = model.to(device=device)

# 超参数
loss_fn = nn.BCELoss()
lr = 0.0001
epochs = 20
save_epoch_step = 1
opt = torch.optim.SGD(model.parameters(), lr=lr)


# 创建保存文件夹
train_idx = 1
while os.path.exists(f'log/train_{train_idx}'):
    train_idx += 1
save_dir = f'log/train_{train_idx}'
os.mkdir(save_dir)



# 导入数据集
traindataset = CiwaBlowholeDataset('data/ciwaquexian/MT_Blowhole/train')
valdataset = CiwaBlowholeDataset('data/ciwaquexian/MT_Blowhole/val')
# testdataset = CiwaBlowholeDataset('data/ciwaquexian/MT_Blowhole/test')
print(len(traindataset), len(valdataset))

# img, label = traindataset[0]
# print(img.shape, label.shape)

traindataloader = DataLoader(dataset=traindataset, batch_size=4 , shuffle= True, num_workers=0,drop_last=True)
valdataloader = DataLoader(dataset=valdataset, batch_size=4 , shuffle= True, num_workers=0,drop_last=True)
# testdataloader = DataLoader(dataset=testdataset, batch_size=1 , shuffle= True, num_workers=0,drop_last=True)

# 开始训练
for i in range(epochs):
    # 训练
    train_loss = []
    model.train()
    for j, data in enumerate(traindataloader):
        # 读数据
        img, label = data
        img = img.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        # 前向传播
        output = model(img)
        # 求损失
        loss = loss_fn(output, label)

        # 清空opt的梯度
        opt.zero_grad()
        # 反向传播
        loss.backward()
        # 优化
        opt.step()

        # 这一个batch的loss
        loss = loss.item() # 转成float
        print(f'epoch{i} batch{j} train_loss={loss:.7f}')
        with open(os.path.join(save_dir, 'train_log.txt'), 'a') as f:
            f.write(f'epoch{i} batch{j} train_loss={loss:.7f}\n')
        train_loss.append(loss)

    # 计算这个epoch的平均loss
    train_loss = np.mean(train_loss) # 这个epoch的平均训练loss
    print(f'epoch{i} avg_train_loss={train_loss:.7f}')
    with open(os.path.join(save_dir, 'train_log.txt'), 'a') as f:
        f.write(f'epoch{i} avg_train_loss={train_loss:.7f}\n#################\n')

    # 验证
    val_loss = []
    model.eval()
    for j, data in enumerate(valdataloader):
        # 读数据
        img, label = data
        img = img.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(img)
            loss = loss_fn(output, label)
            # 这一个batch的loss
            loss = loss.item() # 转成float
            print(f'epoch{i} batch{j} val_loss={loss:.7f}')
            with open(os.path.join(save_dir, 'val_log.txt'), 'a') as f:
                f.write(f'epoch{i} batch{j} val_loss={loss:.7f}\n')
            val_loss.append(loss)

    # 计算这个epoch的平均loss
    val_loss = np.mean(val_loss)
    print(f'epoch{i} avg_val_loss={val_loss:.7f}')
    with open(os.path.join(save_dir, 'val_log.txt'), 'a') as f:
        f.write(f'epoch{i} avg_val_loss={train_loss:.7f}\n#################\n')
    

    # 保存模型
    if (i+1) % save_epoch_step == 0:
        filename = f'epoch{i}_avgtrainloss{train_loss:.7f}_avgvalloss{val_loss:.7f}.pth'
        torch.save(model, os.path.join(save_dir, filename))
        print('model save as ' + os.path.join(save_dir, filename))





