import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from moe_trained import Model
from dataset import MyDataSet
from torch.utils.data import DataLoader
import os

device = 'cuda:0' if torch.cuda.is_available() is not None else 'cpu'
moe_weight_path = './moe_wights/moe_epoch_40000.pth'

batch_size = 64
test_data = MyDataSet('./data', type='val')
test_loader = DataLoader(test_data, batch_size, shuffle=False)

moe = Model(
            input_sizes=[[8, 8], [16, 16], [32, 32]], 
            output_dim=162,
            hidden_dims=[[512, 256],[512, 256]],
            num_experts=8,
            top_p=0.7
        ).to(device)
moe.load_state_dict(torch.load(moe_weight_path))

#开始验证
moe.eval()
cnt = 0
total_loss = 0
total_mse, total_rmse, total_mae = 0, 0, 0
for x_8, x_16, x_32, gt in test_loader:

    # move to device
    x_8, x_16, x_32, gt = x_8.to(device), x_16.to(device), x_32.to(device), gt.to(device)


    output, routing_loss, diversity_loss = moe(x_8, x_16, x_32)

    category_count = test_data.category_counts.to(device)

    output /= category_count
    gt /= category_count


    mse = torch.nn.MSELoss()(output, gt)
    rmse = torch.sqrt(mse)
    mae = torch.nn.L1Loss()(output, gt)

    total_mse += mse.item()
    total_rmse += rmse.item()
    total_mae += mae.item()

    cnt += 1

print(f'   MOE================>MSE:{total_mse / cnt :.4f}, RMSE:{total_rmse / cnt :.4f}, MAE:{total_mae / cnt :.4f}')


