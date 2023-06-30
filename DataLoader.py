import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from icecream import ic

class SensorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name, root_dir, training_length, forecast_window):
        """
        Args:
            csv_file (string): csv文件的路径.
            root_dir (string): 数据文件夹的路径
            training_length (int): 训练序列的长度
            forecast_window (int): 预测序列的长度
        """
        
        # 加载原始数据文件
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()
        self.T = training_length
        self.S = forecast_window

    def __len__(self):
        # 返回传感器的数量
        return len(self.df.groupby(by=["reindexed_id"]))

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, idx):
        
        # Sensors are indexed from 1
        idx = idx+1

        # 用numpy生成一个随机数种子
        # np.random.seed(0)

        # 随机生成一个起始点
        start = np.random.randint(0, len(self.df[self.df["reindexed_id"]==idx]) - self.T - self.S) 

        # 获取传感器编号
        sensor_number = str(self.df[self.df["reindexed_id"]==idx][["sensor_id"]][start:start+1].values.item())

        # 获取输入序列的索引
        index_in = torch.tensor([i for i in range(start, start+self.T)])

        # 获取目标序列的索引
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])

        # 获取输入序列
        _input = torch.tensor(self.df[self.df["reindexed_id"]==idx][["humidity", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start : start + self.T].values)

        # 获取目标序列
        target = torch.tensor(self.df[self.df["reindexed_id"]==idx][["humidity", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start + self.T : start + self.T + self.S].values)

        # 对输入序列进行归一化
        # scalar仅适用于输入序列，以避免缩放值“泄漏”有关目标范围的信息。
        # scalar仅适用于湿度，因为时间戳已经被缩放了
        # scalar的输入/输出形状为：[n_samples，n_features]。
        scaler = self.transform
        scaler.fit(_input[:,0].unsqueeze(-1))
        _input[:,0] = torch.tensor(scaler.transform(_input[:,0].unsqueeze(-1)).squeeze(-1))
        target[:,0] = torch.tensor(scaler.transform(target[:,0].unsqueeze(-1)).squeeze(-1))

        # 保存scalar以便稍后在绘图时反向转换数据
        dump(scaler, 'scalar_item.joblib')

        # 返回索引、输入序列、目标序列和传感器编号
        return index_in, index_tar, _input, target, sensor_number
