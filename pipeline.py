import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.utils.data import Dataset
import logging
import csv
from tqdm import tqdm, trange
import timm

# 定义模型
# from model.swin_transformer import SwinTransformer

# from model.CNN40124 import Net


output_path = "output/VQE2.0"
folder_path = "C:/Users/admin/Desktop/pardata"
# folder_path = "D:/PARRad"
datafile_type = ".h5"
timestampfile_type = ".csv"
# 数据预处理
# 先读取标签 执行read_label程序
from read_label import readlabel

for dirpath, dirnames, filenames in os.walk(folder_path):
    # dirpath为当前文件夹路径，dirnames为当前文件夹中的子文件夹列表，filenames为当前文件夹中的文件列表
    # 判断当前文件夹是否是最后一级子文件夹
    if not dirnames:
        # 遍历当前文件夹中的所有文件
        for filename in filenames:
            # 判断文件类型是否为CSV,同时需要名称为timestamp_speech
            if filename.endswith(timestampfile_type) and "timestamp_speech" in filename:
                # 打印当前文件路径
                datapath = os.path.normpath(os.path.join(dirpath, filename))  # 将路径转换为规范化的形式，斜杠统一
                savepath = os.path.normpath(os.path.join(dirpath, 'output.csv'))
                print("data文件路径：", savepath)
                read_label = readlabel(datapath, savepath)
                read_label.readlabel()

                # print('已读取文件夹', dirpath)

print('已读完!!!!!!!!!!!!!!!!!')

# 保存data和label的.h5文件，即训练时使用的数据文件
from read_data import readdata

for dirpath, dirnames, filenames in os.walk(folder_path):
    # dirpath为当前文件夹路径，dirnames为当前文件夹中的子文件夹列表，filenames为当前文件夹中的文件列表
    # 判断当前文件夹是否是最后一级子文件夹
    if not dirnames:
        # 遍历当前文件夹中的所有文件
        for filename in filenames:
            # 判断文件类型是否为CSV,同时需要名称为timestamp_speech
            if filename.endswith(datafile_type) and "processed_data" in filename:
                # 执行第一步
                datapath = os.path.normpath(os.path.join(dirpath, filename))  # 将路径转换为规范化的形式，斜杠统一 processed_data.h5
                timestamppath = os.path.normpath(os.path.join(dirpath, 'timestamp_speech.csv'))
                savepath = os.path.normpath(os.path.join(dirpath, 'processeddata1.h5'))

                # print("data文件路径：", savepath)
                read_data = readdata(datapath, timestamppath, savepath)
                read_data.readdata1()

                # 执行第二步
                outputpath = os.path.normpath(os.path.join(dirpath, 'output.csv'))
                processeddatapath = os.path.normpath(os.path.join(dirpath, 'processeddata2.h5'))
                read_data = readdata(savepath, outputpath, processeddatapath)
                read_data.readdata2()

                print('已读取文件夹', processeddatapath)
                # value = read_label.readlabel()
print('已保存.h5文件~~~~~~~~~~~~~~~~~~~~~~~~~')

# 定义模型

# from model.VQE import VQE
from model.VQCNN import VQE


# 定义dataset
class Dataset(Dataset):
    def __init__(self, datapath):
        super(Dataset, self).__init__()
        self.datapath = datapath
        self.data = []
        self.label = []
        with h5py.File(self.datapath, 'r') as f:
            for i in f['data'].keys():
                self.data.append(f['data'][i][:])
                self.label.append(f['label'][i][:])

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)


# 读取dataset
# 读取数据文件作为data，读取对应时间戳文件的标签作为label
dataset = []
for dirpath, dirnames, filenames in os.walk(folder_path):
    # dirpath为当前文件夹路径，dirnames为当前文件夹中的子文件夹列表，filenames为当前文件夹中的文件列表
    # 判断当前文件夹是否是最后一级子文件夹
    if not dirnames:
        # 遍历当前文件夹中的所有文件
        for filename in filenames:
            # 判断文件类型是否为CSV,同时需要名称为timestamp_speech
            if filename.endswith(datafile_type) and "processeddata2" in filename:
                datapath = os.path.normpath(os.path.join(dirpath, filename))
                dataset.extend(Dataset(datapath))

print(len(dataset))  # 5202
print('已读取dataset')
print('dataset[0]:')
print(dataset[0][0].shape)  # torch.Size([3, 224, 224])

from sklearn.model_selection import KFold

# 定义K折交叉验证器
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# 遍历每一个折
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    # 定义训练集和验证集
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    # 定义训练集和验证集的数据加载器
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False, drop_last=True)

    # 定义优化器和损失函数
    net = VQE()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.000001, momentum=0.9)
    # 调整学习率
    # # 定义衰减因子和衰减周期
    # decay_factor = 0.8
    # decay_epochs = [5]
    decay_factor = 0.8
    decay_epochs = [2]
    # 定义学习率调整器
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=decay_factor)

    criterion = nn.CrossEntropyLoss()
    latent_loss_weight = 0
    # 训练模型
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 配置日志记录器
    # logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 训练
    num_epochs = 20
    best_acc = 0.0
    net.to(device)
    # 训练过程中记录日志
    logging.info("Start training...")
    train_logs = []  # 用于保存训练日志的列表

    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        train_corrects = 0.0
        train_total = 0.0
        net.train()
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # 形状不对的时候记得改这里，同时记得该下面验证的时候的形状！！！！！！！！！！！！！！！！！！！！！！
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(1)
            #inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
            # 查看输入的形状
            # 删除inputs的第二维度、
            # inputs = inputs.squeeze()
            print(inputs.shape)#(16,1,40,124)
            labels = labels.to(device)
            labels = labels.squeeze()  # 成员函数删除第二维度 16维的向量(16,)
            # labels = labels.squeeze(dim=1)  # 成员函数删除第二维度

            # print(labels.shape)#torch.Size([32, 1])
            # print(labels)
            optimizer.zero_grad()
            outputs, latent_loss = net(inputs)
            latent_loss = latent_loss.mean() * 0.000001
            # latent_loss = latent_loss * latent_loss_weight
            # print(latent_loss)
            recon_loss = criterion(outputs, labels.long())
            # print(recon_loss)
            # lamda = min(max(epoch*0.1,0.1), 1)
            loss = (recon_loss + latent_loss)

            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            preds0 = preds
            labels0 = labels.data
            train_corrects += torch.sum(preds0 == labels.data)
            train_total += labels.size(0)

        train_loss = train_loss / len(train_set)
        train_acc = train_corrects.double() / train_total

        #print('Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, train_loss, train_acc))

        # 在验证集上进行评估
        net.eval()
        test_loss = 0.0
        test_corrects = 0.0
        test_total = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                inputs = inputs.unsqueeze(1)
                inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
                labels = labels.to(device)
                labels = labels.squeeze()  # 成员函数删除第二维度

                outputs, latent_loss = net(inputs)
                latent_loss = latent_loss.mean()
                latent_loss = latent_loss * latent_loss_weight
                recon_loss = criterion(outputs, labels.long())
                loss = recon_loss + latent_loss * latent_loss_weight

                test_loss += loss.item() * inputs.size(0)
                _, preds1 = torch.max(outputs, 1)
                labels1 = labels.data
                test_corrects += torch.sum(preds1 == labels1)
                test_total += labels.size(0)

        test_loss = test_loss / len(val_set)
        test_corrects1 = test_corrects.double()
        test_acc = test_corrects1 / test_total

        # print('Val Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))

        # 打印日志
        loginfo = f"Fold [{fold + 1}/{k}], Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.4f}"
        # log_name = f'train_fold{fold + 1}.log'  # 当前折的日志文件名
        # logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(loginfo)
        print(loginfo)

        # 在每个epoch结束后将训练日志添加到train_logs中
        train_logs.append({
            'Fold': fold + 1,
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train Acc': train_acc.item(),
            'Val Loss': test_loss,
            'Val Acc': test_acc.item()
        })

        # 日志创建文件夹
        if not os.path.exists(output_path):  # 如果文件夹不存在
            os.mkdir(output_path)  # 创建文件夹
            print(f"文件夹{output_path}创建成功")
        # else:
        #     print(f"文件夹{output_path}已经存在")

        # 将train_logs保存为csv文件
        train_logs_name = output_path + '/train_logs_fold' + str(fold + 1) + ".csv"
        with open(train_logs_name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Fold', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])
            writer.writeheader()  # 写入表头
            writer.writerows(train_logs)  # 写入训练日志数据

        # 保存最佳模型
        best_model_path = output_path + '/best_model' + '.pt'
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), best_model_path)
    lr_scheduler.step()

logging.info("Training finished.")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_params = count_parameters(net)
print(f"The number of trainable parameters in the model is {num_params}")
