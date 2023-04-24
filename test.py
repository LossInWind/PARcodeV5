import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2
print('下面是原始数据集的格式')
# file = h5py.File('C:/Users/admin/Desktop/subject_00/captured_data/set000/processed_data.h5', 'r')
#"C:\Users\admin\Desktop\pardata\subject_00\captured_data\set001\processeddata.csv"
with h5py.File('C:/Users/admin/Desktop/pardata/subject_02/captured_data/set001/processed_data.h5', "r") as f:
    for fkey in f.keys():
        print(f[fkey], fkey, f[fkey].name)
        #print(f[fkey], fkey)
    radar_group1 = f["radar"]
    for key1 in radar_group1.keys():
        print(radar_group1[key1], radar_group1[key1].name)
    # print('!!!!!!!')
    radar_group2 = f["radar/TI_xWR68xx"]
    for key2 in radar_group2.keys():
        print(radar_group2[key2], radar_group2[key2].name)
    matrix = f["radar/TI_xWR68xx/mDoppler"][()]  # 获取Dataset对象的值
    # print(matrix.shape)
    # print(matrix.dtype)
    # print(matrix[:])

    # 将矩阵元素缩放到0-255范围内
    arr_scaled = (matrix - matrix.min()) / (matrix.max() - matrix.min()) * 255
    arr_scaled = arr_scaled.astype(np.uint8)

    # 将矩阵转换为灰度图像
    img = cv2.cvtColor(arr_scaled, cv2.COLOR_GRAY2BGR)

    # 显示图像
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #print('!!!!!!!')
    #print(f["radar/TI_xWR14xx/mDoppler"][0])
    #print(f["radar/TI_xWR14xx/mDoppler"][1])
    #matrix = (f["radar/TI_xWR14xx/mDoppler"])
    #print(matrix)
    #plt.imshow(matrix)
    # matrix.show()
    # plt.imshow(matrix)
# 读取数据集
# dataset = file['radar'][:]  # 根据实际情况替换group_name和dataset_name
# print(file['radar'][:])
# print(file['radar'][:].shape)
# 获取数据
# data = dataset[:]

# 关闭文件
# file.close()
f.close()
print('下面是预处理后数据集的格式')

# file = h5py.File('C:/Users/admin/Desktop/subject_00/captured_data/set000/processed_data.h5', 'r')
with h5py.File('C:/Users/admin/Desktop/pardata/subject_02/captured_data/set001/processeddata2.h5', "r") as f:
    for fkey in f.keys():
        print(f[fkey], fkey, f[fkey].name)
        #print(f[fkey], fkey)
    radar_group1 = f["data"]
    for key1 in radar_group1.keys():
        print(radar_group1[key1], radar_group1[key1].name)
    matrix1 = f["data"]['0'][:]
    print(matrix1.shape)
    print(matrix1.dtype)
    print(matrix1)

    # 将矩阵元素缩放到0-255范围内
    arr_scaled = (matrix1 - matrix1.min()) / (matrix1.max() - matrix1.min()) * 255
    arr_scaled = arr_scaled.astype(np.uint8)

    # 将矩阵转换为灰度图像
    img = cv2.cvtColor(arr_scaled, cv2.COLOR_GRAY2BGR)

    # 显示图像
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

f.close()
print('下面是划分训练集后数据集的格式')
with h5py.File('C:/Users/admin/Desktop/data/traindata.h5', "r") as f3:
    for fkey in f3.keys():
        print(f3[fkey], fkey, f3[fkey].name)
    #radar_group1 = f3["train_data"]
    radar_group1 = f3["test_data"]
    for key1 in radar_group1.keys():
        print(radar_group1[key1], radar_group1[key1].name)
    #radar_group2 = f3["train_label"]
    radar_group2 = f3["test_label"]
    for key1 in radar_group2.keys():
        print(radar_group2[key1], radar_group2[key1].name)

    print('!!!!!!')

    matrix3 = f3["train_label/25"]
    matrix4 = f3["train_label/26"]
    print(matrix3[:])
    print(matrix4[:])
    matrix5 = f3["train_data/25"]
    matrix6 = f3["train_data/26"]
    print(matrix5[:])
    print(matrix6[:])

    matrix3 = f3["test_label/25"]
    matrix4 = f3["test_label/26"]
    print(matrix3[:])
    print(matrix4[:])
    matrix5 = f3["test_data/25"]
    matrix6 = f3["test_data/26"]
    print(matrix5[:])
    print(matrix6[:])