"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/16 下午5:06
@Email: 2909981736@qq.com
"""
import ast
import sys
import torch
import argparse
import numpy as np
from utils.set_seed import set_seed
from dataloader.DataLoader import dataLoader

def train(args):
    set_seed(args.seed)


    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bearing Fault Diagnosis")

    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--epochs', type=int, default=10, help='训练的轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--model', type=str, default='PhantomCNN', help='要使用的模型')
    parser.add_argument('--model_params', type=str, default="{'channels': 96, 'kernel_size': 31, 'num_classes': 10}",
                        help='模型超参数字典')
    parser.add_argument('--model_weights', type=str, default=None, help='预训练权重')
    parser.add_argument('--device', type=str, default='cuda:0', help='训练设备')
    parser.add_argument('--snr', type=int, default=None, help='信噪比')
    parser.add_argument('--num_classes', type=int, default=10, help='诊断类别数')
    parser.add_argument('--save_path', type=str, default="log/weights/PhantomCNN.pt", help='权重保存路径')
    parser.add_argument('--data_path', type=str, default="data/CRWU/packaged/12kDE_mix_data.npy", help='数据路径')
    parser.add_argument('--labels_path', type=str, default="data/CRWU/packaged/12kDE_mix_labels.npy", help='标签路径')
    parser.add_argument('--convert_image', type=str, default=None, help='将信号转成图像,不是None就是转换方法名')
    parser.add_argument('--convert_image_path', type=str, default=None, help='转成图像路径')

    args = parser.parse_args()
    args.model_params = ast.literal_eval(args.model_params)

    train(args)
