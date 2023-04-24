"""
这个程序作用是读取数据文件作为data，读取对应时间戳文件的标签作为label

这个程序分成两步：
第一步
输入path1:datapath，读取数据的.h5文件 processed_data.h5
输入path2:timestamppath，读取时间戳的csv文件 timestamp_speech.csv
输出path3:savepath，保存缝合在一起的数据文件路径.h5 processeddata1.h5



第二步
输入path1:savepath，保存缝合在一起的数据文件路径.h5 processeddata.h5
输入path2:outputpath，读取标签的csv文件 output.csv
输出path3:processeddatapath，保存data和label的.h5文件，训练时使用的数据文件 processeddata2.h5

之所以分成两步是因为，之后提特征的矩阵可以修改第一部分或第二部分，比较方便
"""
import os

import h5py
import pandas as pd
import numpy as np
import math
import random

from matplotlib import pyplot as plt


# datapath = 'C:/Users/admin/Desktop/subject_00/captured_data/set000/processed_data.h5'
# timestamppath = 'C:/Users/admin/Desktop/subject_00/captured_data/set000/timestamp_speech.csv'
# savepath = 'C:/Users/admin/Desktop/data/processeddata.h5'
# output = pd.read_csv('C:/Users/admin/Desktop/subject_00/captured_data/set000/output.csv')


class readdata():
    def __init__(self, path1, path2, path3):
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3

    def readdata1(self):
        datapath = self.path1
        timestamppath = self.path2
        savepath = self.path3
        file = h5py.File(datapath, 'r')
        df = pd.read_csv(timestamppath)
        # 将时间戳列转换为日期时间类型
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')

        with h5py.File(datapath, "r") as f1:
            microdoppler2d = f1["radar/TI_xWR68xx/mDoppler"]
            # print(microdoppler2d.shape)
            # 根据时间戳，切割矩阵、获取label、并储存进savepath
            with h5py.File(savepath, 'w') as f2:
                # 创建group
                group_l1 = f2.create_group('data')

                # 创建dataset
                matrix = group_l1.create_dataset('matrix', shape=(0, 128), maxshape=(None, 128), dtype='float32')

                for i in range(len(df) - 1):
                    row0 = df.iloc[0]['timestamp']
                    curr_row = df.iloc[i]['timestamp']
                    next_row = df.iloc[i + 1]['timestamp']
                    time_long = curr_row - row0
                    time_diff = next_row - curr_row

                    totalframe = math.ceil(time_long.total_seconds() / 0.09)  # 向上取整
                    startframe = totalframe + 17
                    #startframe = totalframe
                    # print(startframe)
                    # print(totalframe)
                    # print(i)

                    # endframe = startframe + 40
                    # submatrix = microdoppler2d[:, 413:454]#第414列到第454列
                    if time_diff.total_seconds() >= 3.6:
                        submatrix = microdoppler2d[startframe:startframe + 40, :]
                    else:
                        # 复制最后一行扩充
                        framenum = math.ceil(time_diff.total_seconds() / 0.09)
                        n_repeats = 40 - framenum
                        submatrix0 = microdoppler2d[startframe:startframe + framenum, :]
                        submatrix1 = np.repeat(submatrix0[-1:], n_repeats, axis=0)
                        submatrix = np.concatenate([submatrix0, submatrix1], axis=0)
                    # print(submatrix)
                    # 将新数据追加到数据集中
                    matrix.resize(matrix.shape[0] + submatrix.shape[0],
                                  axis=0)  # 在数据集matrix的最后追加submatrix.shape[0])(第0维)行
                    matrix[-submatrix.shape[0]:, :] = submatrix  # 将submatrix的值赋给matrix的一个切片
                    # filepath = os.path.join('dataimg', str(i))
                    # plt.title(i)
                    # plt.imshow(matrix[:], cmap='jet')
                    # plt.savefig(filepath)  # 保存当前图像为PNG文件
                    # print(matrix[:])
        # 关闭文件
        f1.close()
        f2.close()

    def readdata2(self):

        savepath = self.path1
        outputpath = self.path2
        processeddatapath = self.path3

        output = pd.read_csv(outputpath)
        # 分割保存
        with h5py.File(savepath, "r") as f3:  # 读已经处理好的文件
            microdoppler2d = f3["data/matrix"]
            with h5py.File(processeddatapath, 'w') as f4:

                # 创建一个长度为数据集长度的随机数组，作为索引值
                np.random.seed(42)  # 设置随机种子
                unique_array = []
                while len(unique_array) < (microdoppler2d.shape[0]) / 40:
                    random_num = random.randint(0, round((microdoppler2d.shape[0]) / 40))
                    if random_num not in unique_array:
                        unique_array.append(random_num)
                # print(unique_array)

                # 创建group_l1
                group_l1 = f4.create_group('data')
                group_l2 = f4.create_group('label')

                #
                trainnunm = math.ceil(len(unique_array))
                for i in range(trainnunm - 1):
                    label = output.iloc[i]['label']  # label = output.csv文件中的label列的第i个值
                    data = microdoppler2d[i * 40:i * 40 + 40, :]  # 根据i读取data

                    # #预处理！！！！！！！！！！！！！！！！！！！！！！！
                    # 去除中间的四列
                    data = np.delete(data, np.s_[62:66], axis=1)#此时data的shape为(40,124)
                    #data = data.reshape((1,40,124))
                    # #显示data的数据类型
                    # print(data.dtype)
                    # #显示data的形状
                    # print(data.shape)

                    ##形状转化成[224,224]
                    ## 将data的形状转换为(40, 124)的矩阵
                    # #方法1
                    # old_matrix = data#data的形状为(40,124)
                    # # 将矩阵在第一维复制3遍，变成(3, 40, 124)的矩阵
                    # old_matrix = np.tile(old_matrix, (3, 1, 1))
                    # # 创建一个(3, 224, 224)的全零矩阵
                    # new_matrix = np.zeros((3, 224, 224))
                    # # 将(3, 40, 124)的矩阵复制到(3, 224, 224)的矩阵中
                    # new_matrix[:, 96:96 + 40, 48:48 + 124] = old_matrix[:, :, :]
                    # data = new_matrix
                    # #print(data.shape)
                    #方法2
                    #resize成[3,224,224]
                    ####data = np.resize(data, (1, 224, 224))
                    # print('data的形状为：')
                    # print(data.shape)


                    # filename = str(label) + '_' + str(i) + '.png'  # 生成文件名
                    # filepath = os.path.join('dataimg', filename)
                    # plt.title(i)
                    # plt.imshow(data[:], cmap='jet')
                    # plt.savefig(filepath)  # 保存当前图像为PNG文件


                    # 存入h5文件
                    # 创建dataset，名为
                    data_dataset = group_l1.create_dataset(str(i).encode(), data=data, dtype='float32')
                    label_dataset = group_l2.create_dataset(str(i).encode(), shape=(1, 1), data=label, dtype='int32')
                    # imatrix = f4['data'][str(i)][:]
                    # filepath = os.path.join('dataimg',str(i+100))
                    # plt.title(i)
                    # plt.imshow(imatrix, cmap='jet')
                    # plt.savefig(filepath)  # 保存当前图像为PNG文件
        # 关闭文件
        f3.close()
        f4.close()
