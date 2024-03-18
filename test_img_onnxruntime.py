import onnxruntime as rt
import cv2
import numpy as np
import time



img_path = './input/exp1_num_4727.jpg'

# sess = rt.InferenceSession('./onnx_model/epoch15_avgtrainloss0.0148436_avgvalloss0.0162469.onnx')   # gpu=1.623s/
sess = rt.InferenceSession('./onnx_model/epoch0_avgtrainloss0.2036668_avgvalloss0.2134705.onnx')    # cpu=1.233s

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(input_name, output_name)

# 打印输入节点的名字，以及输入节点的shape
for i in range(len(sess.get_inputs())):
    print(sess.get_inputs()[i].name, sess.get_inputs()[i].shape)

print("----------------")
# 打印输出节点的名字，以及输出节点的shape
for i in range(len(sess.get_outputs())):
    print(sess.get_outputs()[i].name, sess.get_outputs()[i].shape)
    

input_img = cv2.imread(img_path)
input_img = cv2.resize(input_img, (512, 512))
input_img = np.swapaxes(np.swapaxes(input_img, 1, 2), 0, 1)
print(input_img.shape)

t1 = time.time()

output_data = sess.run([output_name], {input_name:[input_img]})
# print(type(output_data))
# output_data = list[array(1,1,512,512)]

t2 = time.time()
print(f'推理耗时{t2-t1:.3f}s')

result = np.swapaxes(np.swapaxes(output_data[0][0], 0, 1), 1, 2)
