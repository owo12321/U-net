import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        ###
        # vgg16下采样

        # (3, 512, 512)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (64, 512, 512)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        # (64, 512, 512) -> out1

        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (64, 256, 256)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (128, 256, 256)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        # (128, 256, 256) -> out2

        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (128, 128, 128)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (256, 128, 128)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (256, 128, 128)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        # (256, 128, 128) -> out3

        self.down4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (256, 64, 64)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (512, 64, 64)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (512, 64, 64)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        # (512, 64, 64) -> out4

        self.down5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (512, 32, 32)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (512, 32, 32)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (512, 32, 32)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        # (512, 32, 32) -> out5

        ###
        # 上采样
        # (512, 32, 32)

        self.upsam = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # (512, 64, 64)
        # concat out4(512, 64, 64)
        # (1024, 64, 64)
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (512, 64, 64)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (512, 64, 64)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # (512, 128, 128)
        )
        # concat out3(256, 128, 128)
        # (768, 128, 128)
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (256, 128, 128)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (256, 128, 128)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # (256, 256, 256)
        )
        # concat out2(128, 256, 256)
        # (384, 256, 256)

        self.up3 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (128, 256, 256)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (128, 256, 256)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # (128, 512, 512)
        )
        # concat out1(64, 512, 512)
        # (192, 512, 512)
        
        self.up4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # (64, 512, 512)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        # (64, 512, 512)

        # 最终输出
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )
        # (1, 512, 512)

    def forward(self, img):
        ###
        # 下采样
        out1 = self.down1(img)      # (64, 512, 512)
        out2 = self.down2(out1)     # (128, 256, 256)
        out3 = self.down3(out2)     # (256, 128, 128)
        out4 = self.down4(out3)     # (512, 64, 64)
        out5 = self.down5(out4)     # (512, 32, 32)

        ###
        # 上采样
        img = self.upsam(out5)           # (512, 64, 64)
        img = torch.cat((img, out4), 1)     # (1024, 64, 64)
        img = self.up1(img)                 # (512, 128, 128)

        img = torch.cat((img, out3), 1)     # (768, 128, 128)
        img = self.up2(img)                 # (256, 256, 256)
        
        img = torch.cat((img, out2), 1)     # (384, 256, 256)
        img = self.up3(img)                 # (128, 512, 512)

        img = torch.cat((img, out1), 1)     # (192, 512, 512)
        img = self.up4(img)                 # (64, 512, 512)

        res = self.final(img)               # (1, 512, 512)

        return res