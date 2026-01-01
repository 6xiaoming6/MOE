import os.path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model_v1 import Model
from dataset import MyDataSet
from torch.utils.data import DataLoader
from datetime import datetime
import logging

#训练时保存日志和模型的文件夹会加上这个名字以区分
train_name = "normal_data"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 64
epochs = 1000
save_freq, val_freq, print_freq = 100, 10, 10
save_root_dir = './checkpoints'

if not os.path.isdir(save_root_dir):
    os.makedirs(save_root_dir)

train_data, val_data = MyDataSet(root_dir='./data',normalize=True), MyDataSet(root_dir='./data', normalize=True, type='val')
train_loader, val_loader = DataLoader(train_data, batch_size, False), DataLoader(val_data, batch_size, False)
hidden_dim, num_experts, top_k = 1024, 8, 3

net = Model(
            input_sizes=[[8, 8], [16, 16], [32, 32]], 
            output_dim=162,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k
        ).to(device)

learning_rate = 5e-4
lambdas = [1.0, 0.05]
loss_fn = torch.nn.MSELoss()
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
optimizer_name = type(optimizer).__name__

start_train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') #记录此次训练开始的时间，作为路径保存到模型中

model_save_dir = f'{save_root_dir}/{start_train_time}_{train_name}/model' #模型的权重最终保存的路径
log_save_dir = f'{save_root_dir}/{start_train_time}_{train_name}/log' #日志文件最终保存的路径
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
if not os.path.exists(log_save_dir):
    os.makedirs(log_save_dir)

def setup_logger():
    """配置日志系统"""
    # 创建自定义记录器
    logger = logging.getLogger('trainer')
    logger.setLevel(logging.DEBUG)

    # 清除现有的处理器
    logger.handlers.clear()

    # 阻止日志传播到根记录器（关键！）
    logger.propagate = False

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 简洁的控制台格式
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # 1. val.log 文件处理器 - 只记录DEBUG级别（验证详情）
    val_file_handler = logging.FileHandler(
        filename=os.path.join(log_save_dir, 'val.log'),
        encoding='utf-8',
        mode='w'  # 每次训练覆盖旧文件
    )
    val_file_handler.setLevel(logging.DEBUG)
    val_file_handler.setFormatter(formatter)

    # 添加过滤器，只允许DEBUG级别的验证日志
    def val_debug_filter(record):
        # 只记录验证相关的DEBUG日志
        return record.levelno == logging.DEBUG and 'MSE:' in record.getMessage()

    val_file_handler.addFilter(val_debug_filter)

    # 2. train.log 文件处理器 - 记录所有INFO及以上级别
    train_file_handler = logging.FileHandler(
        filename=os.path.join(log_save_dir, 'train.log'),
        encoding='utf-8',
        mode='w'
    )
    train_file_handler.setLevel(logging.INFO)
    train_file_handler.setFormatter(formatter)

    # 3. 控制台处理器 - 只记录INFO及以上级别，使用简洁格式
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # 添加处理器到日志记录器
    logger.addHandler(train_file_handler)
    logger.addHandler(val_file_handler)
    logger.addHandler(console_handler)

    return logger

def save_config():
    train_logger.info(f'hidden_num: {hidden_dim}')
    train_logger.info(f'num_experts: {num_experts}')
    train_logger.info(f'top-k: {top_k}')
    train_logger.info(f'device: {device}')
    train_logger.info(f'batch_size: {batch_size}')
    train_logger.info(f'epochs: {epochs}')
    train_logger.info(f'learning_rate: {learning_rate}')
    train_logger.info(f'lambdas: {lambdas}')
    train_logger.info(f'loss_fn: MSELoss')
    train_logger.info(f'optimizer: {optimizer_name}')


train_logger = setup_logger()
save_config()

train_logger.info('开始训练')
for cur_epoch in range(epochs):
    cnt = 0
    total_loss = 0
    total_mse, total_rmse, total_mae =  0, 0, 0
    origin_total_mse, origin_total_rmse, origin_total_mae =  0, 0, 0
    for x_8, x_16, x_32, gt, gt_origin, vmin, vmax in train_loader:
        optimizer.zero_grad() #计算前梯度清零

        #前向计算
        x_8, x_16, x_32, gt, gt_origin, vmin, vmax = x_8.to(device), x_16.to(device), x_32.to(device), gt.to(device), gt_origin.to(device), vmin.to(device), vmax.to(device)

        output, load_balance_loss  = net(x_8, x_16, x_32)
        output, load_balance_loss = output.to(device), load_balance_loss.to(device)

        category_count = train_data.category_counts.to(device)
        projection_mask_matrix = train_data.projection_mask_matrix.to(device)
        output_origin = output * (vmax.view(-1,1) - vmin.view(-1,1)) + vmin.view(-1,1)

        # output /= category_count
        # gt /= category_count

        mse = loss_fn(output, gt)
        
        loss = lambdas[0] * mse + lambdas[1] * load_balance_loss

        total_mse += mse.item()
        total_rmse += torch.sqrt(mse).item()
        total_mae += torch.nn.L1Loss()(output, gt).item()
        total_loss += loss.item()

        with torch.no_grad():
            output_origin /= category_count
            gt_origin /= category_count

            origin_total_mse += loss_fn(output_origin, gt_origin).item()
            origin_total_rmse += torch.sqrt(loss_fn(output_origin, gt_origin)).item()
            origin_total_mae += torch.nn.L1Loss()(output_origin, gt_origin).item()

        cnt += 1

        #反向传播
        loss.backward()
        #更细参数
        optimizer.step()

    #定期保存模型参数
    if (cur_epoch + 1) % save_freq == 0:
        model_save_path = os.path.join(model_save_dir, f'model_{cur_epoch+1}.pth')
        opt_save_path = os.path.join(model_save_dir, f'opt_{cur_epoch+1}.pth')

        torch.save(net.state_dict(), model_save_path)
        torch.save(optimizer.state_dict(), opt_save_path)

        train_logger.info(f'[Epoch {cur_epoch + 1}/{epochs}] --- model save to: {model_save_path}')

    if(cur_epoch + 1) % print_freq == 0:
        train_logger.info(
            f'[Epoch {cur_epoch + 1}/{epochs}] --- Loss: {total_loss / cnt :.6f}, MSE: {total_mse / cnt :.6f}, RMSE: {total_rmse / cnt :.6f}, MAE: {total_mae / cnt :.6f}')
        train_logger.info(
            f'[Epoch {cur_epoch + 1}/{epochs}] --- Origin MSE: {origin_total_mse / cnt :.6f}, Origin RMSE: {origin_total_rmse / cnt :.6f}, Origin MAE: {origin_total_mae / cnt :.6f}')

    #定期验证模型效果
    if(cur_epoch + 1) % val_freq == 0:
        net.eval()
        cnt = 0
        origin_total_mse, origin_total_rmse, origin_total_mae = 0, 0, 0
        for x_8, x_16, x_32, gt, gt_origin, vmin, vmax in val_loader:
            # move to device
            x_8, x_16, x_32, gt, gt_origin, vmin, vmax = x_8.to(device), x_16.to(device), x_32.to(device), gt.to(device), gt_origin.to(device), vmin.to(device), vmax.to(device)

            output, load_balance_loss = net(x_8, x_16, x_32)

            category_count = val_data.category_counts.to(device)
            projection_mask_matrix = train_data.projection_mask_matrix.to(device)

            origin_output = output * (vmax.view(-1, 1) - vmin.view(-1, 1)) + vmin.view(-1, 1)

            gt_origin /= category_count
            origin_output /= category_count

            mse = loss_fn(origin_output, gt_origin)
            rmse = torch.sqrt(mse)
            mae = torch.nn.L1Loss()(origin_output, gt_origin)

            origin_total_mse += mse.item()
            origin_total_rmse += rmse.item()
            origin_total_mae += mae.item()

            cnt += 1

        train_logger.debug(f'[Epoch {cur_epoch + 1}/{epochs}] --- MSE:{origin_total_mse / cnt :.4f}, RMSE:{origin_total_rmse / cnt :.4f}, MAE:{origin_total_mae / cnt :.4f}')









