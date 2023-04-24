import os

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt

print('下面是预处理后数据集的格式')

# file = h5py.File('C:/Users/admin/Desktop/subject_00/captured_data/set000/processed_data.h5', 'r')
with h5py.File('D:/PARRad/subject_00/captured_data/set000/processeddata2.h5', "r") as f:
    label = f['label']

    for i in range(len(label)):
        ilabel = label[str(i)]
        imatrix = f['data'][str(i)][:]
        plt.title(ilabel[0][0])
        plt.imshow(imatrix, cmap='jet')
        # 添加横纵坐标
        plt.xticks(range(len(imatrix[0])))
        plt.yticks(range(len(imatrix)))
        filename = str(ilabel[0][0]) + '_' + str(i) + '.png'  # 生成文件名
        # 生成完整的文件路径
        filepath = os.path.join('../dataimg', filename)
        print(imatrix.shape)
        plt.savefig(filepath)  # 保存当前图像为PNG文件
        plt.show()
        print(ilabel[0][0])

    startframe = 0

f.close()