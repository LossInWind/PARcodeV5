"""
这个程序主要用来把标签读出来，并且判断是否有小于3.7s的动作
输入readpath，例子：readpath = pd.read_csv('C:/Users/admin/Desktop/subject_00/captured_data/set000/timestamp_speech.csv')
输出savepath，作为读取标签的csv文件 output.csv

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 读取CSV文件，假设时间戳列名为'timestamp'

#readpath = pd.read_csv('C:/Users/admin/Desktop/subject_00/captured_data/set000/timestamp_speech.csv')


class readlabel():
    def __init__(self, readpath,savepath):
        self.readpath = readpath
        self.savepath = savepath

    def readlabel(self):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(self.readpath)

        # 将时间戳列转换为日期时间类型
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')

        # 将日期时间类型分解为年、月、日、小时、分钟和秒
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        df['microsecond'] = df['timestamp'].dt.microsecond
        # 获取command列
        command_col = df['command']

        # # 显示结果
        # print(df.head())
        # print(df['hour'])
        # print(df['microsecond'])

        # 初始化结果列表
        result = [0] * len(df)
        label = [0] * len(df)
        label_last = None  # 给label_last赋一个默认值
        #label.iloc[-1] = None  # 在这里将label_last初始化为None
        # 遍历所有行，计算相邻两行之间的时间差
        for i in range(len(df) - 1):

            # prev_row = df.iloc[i-1]['timestamp']
            curr_row = df.iloc[i]['timestamp']
            next_row = df.iloc[i + 1]['timestamp']
            time_diff = next_row - curr_row
            if time_diff.total_seconds() >= 3.7:
                result[i] = 1
            elif time_diff.total_seconds() < 3.7:
                result[i] = 0
            # 处理最后一行
            if (df.iloc[-1]['timestamp'] - df.iloc[-2]['timestamp']).total_seconds() >= 3.7:
                result[-1] = 1
            else:
                result[-1] = 0

            # 标记样本标签
            if command_col.iloc[i] == 'walk to chair' or command_col.iloc[i] == 'walk to room' or command_col.iloc[
                i] == 'walk to bed':
                label[i] = 1
            elif command_col.iloc[i] == 'sit down on chair' or command_col.iloc[i] == 'sit down on bed':
                label[i] = 2
            elif command_col.iloc[i] == 'stand up from chair' or command_col.iloc[i] == 'stand up from bed':
                label[i] = 3
            elif command_col.iloc[i] == 'fall on floor of room':
                label[i] = 4
            elif command_col.iloc[i] == 'stand up from floor of room':
                label[i] = 5
            elif command_col.iloc[i] == 'get in  bed':
                label[i] = 6
            elif command_col.iloc[i] == 'lie in bed':
                label[i] = 7
            elif command_col.iloc[i] == 'roll in bed':
                label[i] = 8
            elif command_col.iloc[i] == 'sit in bed':
                label[i] = 9
            elif command_col.iloc[i] == 'get out  bed':
                label[i] = 10
            # 处理最后一行
            last_command = command_col.iloc[-1]  # 获取最后一行的值
            if last_command == 'walk to chair' or last_command == 'walk to room' or last_command == 'walk to bed':
                label_last = 1
            elif last_command == 'sit down on chair':
                label_last = 2
            elif last_command == 'stand up from chair' or last_command == 'stand up from bed':
                label_last = 3
            elif last_command == 'fall on floor of room':
                label_last = 4
            elif last_command == 'stand up from floor of room':
                label_last = 5
            elif last_command == 'get in  bed':
                label_last = 6
            elif last_command == 'lie in bed':
                label_last = 7
            elif last_command == 'roll in bed':
                label_last = 8
            elif last_command == 'sit in bed':
                label_last = 9
            elif last_command == 'get out  bed':
                label_last = 10

            label = pd.Series(label)
            label.iloc[-1] = label_last

        # 将结果列表添加为新列到DataFrame
        df['result'] = result
        df['label'] = label
        # 显示结果
        #print(df['result'])
        #print(df['label'])
        # 将DataFrame保存为新的CSV文件，假设文件名为'output.csv'
        #df.to_csv('C:/Users/admin/Desktop/subject_00/captured_data/set000/output.csv', index=False)
        df.to_csv(self.savepath, index=False)
        #return


