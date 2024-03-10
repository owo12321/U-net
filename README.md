# U-net
这是一个U-net的pytorch实现  
![image](./Unet.jpg)  
下采样部分改为使用vgg16，且上下采样拼接的部分尺寸一致，不用裁剪  
输入为(3,512,512)，输出为(512,512)  
数据集：Saliency of magnetic tile surface defects  
https://www.cvmart.net/dataSets/detail/278