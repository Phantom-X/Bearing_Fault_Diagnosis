"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/30 下午7:34
@Email: 2909981736@qq.com
"""
import torch
import os.path
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from data_preprocess.SDP.SDP import SDP
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.set_seed import set_seed

set_seed(42)


class dataLoader:
    def __init__(self):
        super().__init__()

    @classmethod
    def signal(cls, data, labels, batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_ratio,
                                                                            random_state=42)
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                                          train_size=train_ratio,
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
        val_num = len(val_dataset)
        train_num = len(train_dataset)
        test_num = len(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_steps = len(train_loader)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_steps, train_num, val_num, test_num, train_loader, val_loader, test_loader

    @classmethod
    def sdp(cls, data, labels, save_path=None, batch_size=32, train_ratio=0.8, val_ratio=0.1,
            test_ratio=0.1):

        if len(glob(save_path + "/*")) == 0:
            count = 0
            for one_data, label in tqdm(zip(data, labels), total=len(data), desc="Generate SDP image"):
                one_data_savepath = f"{save_path}/{int(label)}"
                os.makedirs(one_data_savepath, exist_ok=True)
                SDP(one_data, savepath=f"{one_data_savepath}/{int(label)}_{count}")
                count += 1

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])

        tensor_images = []
        tensor_labels = []
        for one_data in glob(f"{save_path}/*/*.png"):
            image = Image.open(one_data)
            image = image.convert("RGB")
            transformed_image = transform(image)
            label = int(one_data.split("/")[-2])
            tensor_images.append(transformed_image)
            tensor_labels.append(label)

        image_tensor = torch.stack(tensor_images)
        # print(image_tensor.shape)
        label_tensor = torch.tensor(tensor_labels)
        # print(label_tensor.shape)
        dataset = TensorDataset(image_tensor, label_tensor)

        train_size = int(train_ratio * len(dataset))
        test_size = int(test_ratio * len(dataset))
        val_size = len(dataset) - train_size - test_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                 [train_size, val_size, test_size])
        val_num = len(val_dataset)
        train_num = len(train_dataset)
        test_num = len(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_steps = len(train_loader)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_steps, train_num, val_num, test_num, train_loader, val_loader, test_loader


if __name__ == '__main__':
    data = np.load("../data/CRWU/packaged/12kDE_mix_data.npy")
    labels = np.load("../data/CRWU/packaged/12kDE_mix_labels.npy")
    dataLoader.sdp(data, labels, save_path="../data/SDP/12kDE_mix_data")
