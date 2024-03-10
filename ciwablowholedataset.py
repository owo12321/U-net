import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import os


class CiwaBlowholeDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.filenames = os.listdir(self.path)
    def __len__(self):
        return len(self.filenames)//2
    def __getitem__(self, index):
        datafilename = self.filenames[index*2]
        labelfilename = self.filenames[index*2+1]
        
        data = cv2.resize(cv2.imread(os.path.join(self.path,datafilename)), (512, 512))
        # 调整通道顺序
        data = np.swapaxes(np.swapaxes(data, 0, 2), 1, 2)

        # 标签通道数为1
        label = cv2.resize(cv2.imread(os.path.join(self.path,labelfilename), 0), (512, 512))[np.newaxis,:,:]
        ret, label = cv2.threshold(label, 175, 1, cv2.THRESH_BINARY)    # 把标签变成0到1

        return data, label