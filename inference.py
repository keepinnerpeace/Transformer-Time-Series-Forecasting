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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def inference(path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model):

    # 将device转换为torch.device类型
    device = torch.device(device)
    
    # 创建Transformer模型实例，将其转换为double类型，并将其移动到指定的device上
    model = Transformer().double().to(device)
    
    # 加载预训练模型
    model.load_state_dict(torch.load(path_to_save_model+best_model))
    
    # 定义损失函数为均方误差
    criterion = torch.nn.MSELoss()

    # 初始化验证集损失为0
    val_loss = 0
    
    # 关闭梯度计算
    with torch.no_grad():
        
        # 将模型设置为评估模式
        model.eval()
        
        # 对于25个plot
        for plot in range(25):


            for index_in, index_tar, _input, target, sensor_number in dataloader:
                
                # starting from 1 so that src matches with target, but has same length as when training
                # 将输入数据的维度从 (batch_size, seq_len, feature_dim) 转换为 (seq_len, batch_size, feature_dim)
                src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
                # 将目标数据的维度从 (batch_size, seq_len, feature_dim) 转换为 (seq_len, batch_size, feature_dim)
                target = target.permute(1,0,2).double().to(device) # t48 - t59

                # 将模型下一个输入设置为当前输入数据
                next_input_model = src
                # 初始化所有预测结果为空列表
                all_predictions = []

                for i in range(forecast_window - 1):
                    
                    # 用模型预测下一个时间步的值
                    prediction = model(next_input_model, device) # 47,1,1: t2' - t48'

                    if all_predictions == []:
                        all_predictions = prediction # 47,1,1: t2' - t48'
                    else:
                        # 将预测结果拼接到 all_predictions 中
                        all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'


                    # 获取当前时间步之后的所有位置编码
                    pos_encoding_old_vals = src[i+1:, :, 1:] # 46, 1, 6, pop 位置编码的第一个值: t2 -- t47
                    # 获取最后一个预测值的位置编码
                    pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1) # 1, 1, 6, 添加最后一个预测值的位置编码: t48
                    # 将所有位置编码与预测值匹配
                    pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6: t2 -- t48 的位置编码
                    
                    # 将当前时间步之后的输入数据和最后一个预测值拼接在一起，作为下一个时间步的输入数据
                    next_input_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1,:,:].unsqueeze(0))) #t2 -- t47, t48'
                    # 将位置编码拼接在一起，作为下一个时间步的输入数据
                    next_input_model = torch.cat((next_input_model, pos_encodings), dim = 2) # 47, 1, 7 input for next round

                true = torch.cat((src[1:,:,0],target[:-1,:,0]))
                loss = criterion(true, all_predictions[:,:,0])
                val_loss += loss
            
            val_loss = val_loss/10 # 将验证集的损失除以10，得到平均损失
            scaler = load('scalar_item.joblib') # 加载标量
            # 将输入数据的湿度值转换为原始值
            src_humidity = scaler.inverse_transform(src[:,:,0].cpu())
            # 将目标数据的湿度值转换为原始值
            target_humidity = scaler.inverse_transform(target[:,:,0].cpu())
            # 将所有预测结果的湿度值转换为原始值
            prediction_humidity = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())
            # 绘制预测结果图像
            plot_prediction(plot, path_to_save_predictions, src_humidity, target_humidity, prediction_humidity, sensor_number, index_in, index_tar)


        logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")