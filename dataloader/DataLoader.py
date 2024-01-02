"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/30 下午7:34
@Email: 2909981736@qq.com
"""
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch


class dataLoader():
    def __init__(self, data_type="signal"):
        super().__init__()
        self.loader = None
        if data_type == "signal":
            self.loader = signal
        elif data_type == "image":
            self.loader = image

    def __call__(self, data, labels, batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        train_dataset, val_dataset, test_dataset = self.loader(data, labels, train_ratio=train_ratio,
                                                               val_ratio=val_ratio, test_ratio=test_ratio)
        self.train_num = len(train_dataset)
        self.val_num = len(val_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


def signal(data, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_ratio,
                                                                        random_state=42)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, train_size=train_ratio,
                                                                      random_state=42)
    # 转tensor
    train_data = torch.from_numpy(train_data).float()
    train_data = train_data.unsqueeze(1)
    train_labels = torch.from_numpy(train_labels).long()

    val_data = torch.from_numpy(val_data).float()
    val_data = val_data.unsqueeze(1)
    val_labels = torch.from_numpy(val_labels).long()

    test_data = torch.from_numpy(test_data).float()
    test_data = test_data.unsqueeze(1)
    test_labels = torch.from_numpy(test_labels).long()

    # 创建训练集、验证集和测试集的数据集对象
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    return train_dataset, val_dataset, test_dataset


def image(data, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    return TensorDataset(), TensorDataset(), TensorDataset()
