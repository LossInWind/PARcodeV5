import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import os.path
# 读取每个折的训练日志，将其存储到一个列表中
train_logs_list = []
# 一些参数
k = 5
num_epochs = 20
output_path = "output/VQE2.0"
merged_train_logs_name = output_path + '/merged_train_logs.csv'

if os.path.exists(merged_train_logs_name):
    print("日志存在，读取中。。。")
    # 读取合并后的训练日志，并绘制Accuracy折线图
    df = pd.read_csv(merged_train_logs_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    for fold in range(k):
        fold_df = df[df['Fold'] == fold + 1]
        ax.plot(fold_df['Epoch'], fold_df['Train Acc'], label=f'Fold {fold + 1} Train Acc')
        ax.plot(fold_df['Epoch'], fold_df['Val Acc'], label=f'Fold {fold + 1} Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy vs. Epoch')
    ax.legend()
    plt.show()

    # 读取合并后的训练日志，并绘制Loss折线图
    df = pd.read_csv(merged_train_logs_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    for fold in range(k):
        fold_df = df[df['Fold'] == fold + 1]
        ax.plot(fold_df['Epoch'], fold_df['Train Loss'], label=f'Fold {fold + 1} Train Loss')
        ax.plot(fold_df['Epoch'], fold_df['Val Loss'], label=f'Fold {fold + 1} Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Loss vs. Epoch')
    ax.legend()
    plt.show()

else:
    print("日志不存在")
    for fold in range(k):
        train_logs_name = output_path + '/train_logs_fold' + str(fold + 1) + ".csv"
        print(f'Reading train logs from {train_logs_name}...')
        with open(train_logs_name, 'r') as f:
            reader = csv.DictReader(f)
            train_logs = [row for row in reader]
            train_logs_list.append(train_logs)
        print(f'Read {len(train_logs)} lines from {train_logs_name}.')

    # # 将所有折的训练日志合并到一个列表中
    merged_train_logs = []
    for fold in range(k):
        epoch_logs = {'Fold': fold + 1}
        for epoch in range(num_epochs):
            epoch_logs.update(train_logs_list[fold][epoch])
            epoch_logs = {'Epoch': epoch + 1}
            merged_train_logs.append(epoch_logs)
    print(f'Read {len(merged_train_logs)} lines from merged_train_logs.')

    # 删除每个折的单独csv文件
    for fold in range(k):
        train_logs_name = output_path + '/train_logs_fold' + str(fold + 1) + ".csv"
        os.remove(train_logs_name)
        print(f'Removing {train_logs_name}...')

    # 将合并后的训练日志保存为csv文件

    with open(merged_train_logs_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Epoch', 'Fold', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])
        writer.writeheader()  # 写入表头
        writer.writerows(merged_train_logs)  # 写入训练日志数据

    # 读取合并后的训练日志，并绘制Accuracy折线图
    df = pd.read_csv(merged_train_logs_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    for fold in range(k):
        fold_df = df[df['Fold'] == fold + 1]
        ax.plot(fold_df['Epoch'], fold_df['Train Acc'], label=f'Fold {fold + 1} Train Acc')
        ax.plot(fold_df['Epoch'], fold_df['Val Acc'], label=f'Fold {fold + 1} Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy vs. Epoch')
    ax.legend()
    plt.show()

    # 读取合并后的训练日志，并绘制Loss折线图
    df = pd.read_csv(merged_train_logs_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    for fold in range(k):
        fold_df = df[df['Fold'] == fold + 1]
        ax.plot(fold_df['Epoch'], fold_df['Train Loss'], label=f'Fold {fold + 1} Train Loss')
        ax.plot(fold_df['Epoch'], fold_df['Val Loss'], label=f'Fold {fold + 1} Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Loss vs. Epoch')
    ax.legend()
    plt.show()



