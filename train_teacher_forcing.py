from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import SensorDataset
import logging
import time # debugging
from plot import *
from helpers import *
from joblib import load
from icecream import ic
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def transformer(dataloader, EPOCH, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):

    # 将device转换为torch.device类型
    device = torch.device(device)

    # 实例化Transformer模型，并将其转换为双精度类型，然后将其移动到指定的设备上
    model = Transformer().double().to(device)

    # 实例化Adam优化器，将其用于模型参数的优化
    optimizer = torch.optim.Adam(model.parameters())

    # 实例化均方误差损失函数
    criterion = torch.nn.MSELoss()

    # 初始化最佳模型的名称为空字符串
    best_model = ""

    # 初始化最小训练损失为正无穷
    min_train_loss = float('inf')


    for epoch in range(EPOCH + 1):

        train_loss = 0
        val_loss = 0

        ## TRAIN -- TEACHER FORCING
        model.train() # 将模型设置为训练模式
        for index_in, index_tar, _input, target, sensor_number in dataloader: # 遍历数据集中的每个数据

            optimizer.zero_grad() # 将优化器的梯度清零

            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]

            src = _input.permute(1,0,2).double().to(device)[:-1,:,:] # 将输入数据的维度从[batch, input_length, feature]转换为[input_length, batch, feature]，并将其转换为双精度类型，然后将其移动到指定的设备上
            target = _input.permute(1,0,2).double().to(device)[1:,:,:] # 将输入数据的维度从[batch, input_length, feature]转换为[input_length, batch, feature]，并将其转换为双精度类型，然后将其移动到指定的设备上，再将其向前移动一个时间步
            prediction = model(src, device) # 使用模型进行预测
            loss = criterion(prediction, target[:,:,0].unsqueeze(-1)) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 优化模型参数
            # scheduler.step(loss.detach().item())
            train_loss += loss.detach().item() # 累加训练损失


        if train_loss < min_train_loss:
            # 如果当前训练损失小于最小训练损失，则保存当前模型参数和优化器状态，并更新最小训练损失和最佳模型名称
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}.pth"


        if epoch % 100 == 0: # 每100个epoch，绘制1步预测图

            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            scaler = load('scalar_item.joblib')
            src_humidity = scaler.inverse_transform(src[:,:,0].cpu()) # 将输入数据的湿度特征还原为原始值
            target_humidity = scaler.inverse_transform(target[:,:,0].cpu()) # 将目标数据的湿度特征还原为原始值
            prediction_humidity = scaler.inverse_transform(prediction[:,:,0].detach().cpu().numpy()) # 将预测数据的湿度特征还原为原始值
            plot_training(epoch, path_to_save_predictions, src_humidity, prediction_humidity, sensor_number, index_in, index_tar) # 绘制1步预测图


        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
        
    plot_loss(path_to_save_loss, train=True)
    return best_model