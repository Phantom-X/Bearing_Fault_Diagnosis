"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/16 下午5:06
@Email: 2909981736@qq.com
"""
import os
import ast
import sys
import torch
from torch import nn, optim
import argparse
from tqdm import tqdm
import numpy as np
from utils.set_seed import set_seed
from dataloader.DataLoader import dataLoader
from utils.add_noise import batch_add_noise
from utils.dynamic_load import get_class_in_module


def train(args):
    set_seed(args.seed)
    data = np.load(args.data_path)
    data = batch_add_noise(data, args.snr)
    print(data.shape)
    labels = np.load(args.labels_path)
    print(labels.shape)
    if args.convert_image is None:
        train_steps, train_num, val_num, test_num, train_loader, val_loader, test_loader = dataLoader.signal(data,
                                                                                                             labels,
                                                                                                             batch_size=args.batch_size,
                                                                                                             train_ratio=0.8,
                                                                                                             val_ratio=0.1,
                                                                                                             test_ratio=0.1)
    elif args.convert_image == "SDP":
        train_steps, train_num, val_num, test_num, train_loader, val_loader, test_loader = dataLoader.sdp(data, labels,
                                                                                                          save_path=args.convert_image_path,
                                                                                                          batch_size=args.batch_size,
                                                                                                          train_ratio=0.8,
                                                                                                          val_ratio=0.1,
                                                                                                          test_ratio=0.1)
    else:
        raise ValueError

    Model = get_class_in_module(args.model, os.path.join("models", args.model))
    model = Model(**args.model_params)
    if args.model_weights is not None:
        model.load_state_dict(torch.load(args.model_weights))
    model.to(args.device)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # , weight_decay=0.001

    best_acc = 0.0
    for epoch in range(args.epochs):
        # train
        model.train()
        running_loss = 0.0
        acc_train = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, train_data in enumerate(train_bar):
            data, labels = train_data
            optimizer.zero_grad()
            outputs = model(data.to(args.device))
            predict_train = torch.max(outputs, dim=1)[1]
            acc_train += torch.eq(predict_train, labels.to(args.device)).sum().item()
            labels = torch.nn.functional.one_hot(labels, num_classes=args.num_classes).float()
            loss = loss_function(outputs, labels.to(args.device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{args.epochs}] loss:{loss:.3f}"

        # val
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                data, labels = val_data
                outputs = model(data.to(args.device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(args.device)).sum().item()
                val_bar.desc = f"val epoch[{epoch + 1}/{args.epochs}]"

        train_accurate = acc_train / train_num
        val_accurate = acc / val_num
        print(
            f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f} train_accuracy: {train_accurate:.3f} val_accuracy: {val_accurate:.3f}')

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), args.save_path)

    print('val best acc:', best_acc)
    print("Finished Training")

    # 在测试集上评估模型
    test_correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(args.device)
            labels = labels.to(args.device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / test_num

    print(f"Test Accuracy: {test_accuracy:.4f}")


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
    parser.add_argument('--save_path', type=str, default="log/weights/PhantomCNN.pth", help='权重保存路径')
    parser.add_argument('--data_path', type=str, default="data/CRWU/packaged/12kDE_mix_data.npy", help='数据路径')
    parser.add_argument('--labels_path', type=str, default="data/CRWU/packaged/12kDE_mix_labels.npy", help='标签路径')
    parser.add_argument('--convert_image', type=str, default=None, help='将信号转成图像,None 或者 转换方法')
    parser.add_argument('--convert_image_path', type=str, default=None, help='转成图像路径')

    args = parser.parse_args()
    args.model_params = ast.literal_eval(args.model_params)

    train(args)
