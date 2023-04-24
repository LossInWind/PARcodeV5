"""
这个程序作用是读取data和label转化成numpy格式，方便输入进dataloader里


"""

import os

import h5py
import torch


class loaddata():
    def __init__(self, folder_path):
        super(loaddata, self).__init__()
        self.folder_path = folder_path
        self.data = []
        self.label = []

        datafile_type = ".h5"
        timestampfile_type = ".csv"
        # 遍历文件夹及其子文件夹
        for dirpath, dirnames, filenames in os.walk(folder_path):
            # dirpath为当前文件夹路径，dirnames为当前文件夹中的子文件夹列表，filenames为当前文件夹中的文件列表

            # 判断当前文件夹是否是最后一级子文件夹
            if not dirnames:
                # 遍历当前文件夹中的所有文件
                for filename in filenames:
                    # 判断文件类型是否为file_type,同时需要名称为load_data
                    if filename.endswith(datafile_type) and "load_data" in filename:
                        # 打印当前文件路径
                        datapath = os.path.normpath(os.path.join(dirpath, filename))  # 将路径转换为规范化的形式，斜杠统一

                        #print("data文件路径：", datapath)
                        with h5py.File(datapath, 'r') as f:
                            for i in f['data'].keys():  # 遍历读取
                                self.data.append(f['data'][i][:])
                                self.label.append(f['label'][i][:])


    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)

