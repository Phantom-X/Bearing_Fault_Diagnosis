"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/16 下午5:10
@Email: 2909981736@qq.com
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random
import sys
from utils.set_seed import set_seed
from models.PhantomCNN import PhantomCNN
from utils.add_noise import batch_add_noise
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# 超参数设置
# 随机数
set_seed(42)
# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 信噪比
snr_db = -4
# 模型参数
model_params = {'channels': 256, 'kernel_size': 63, 'dilation': [1, 2, 5, 1, 2], 'num_classes': 3, 'phantomlayer_num':5}
# 模型权重
model_weights = "log/weights/PhantomCNN_matrix_1.pth"
# 数据路径
data_path = "data/UPB/packaged/naturally_damaged_N15_M01_F10_data.npy"
# 数据标签路径
labels_path = "data/UPB/packaged/naturally_damaged_N15_M01_F10_labels.npy"

model = PhantomCNN(**model_params)
model.load_state_dict(torch.load(model_weights))
model.to(device)
model.eval()

# 加载原始数据
data = np.load(data_path)
data = batch_add_noise(data, snr_db)
labels = np.load(labels_path)

# 划分数据集,获取测试集
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, random_state=42)
test_data = torch.from_numpy(test_data).float()
test_data = test_data.unsqueeze(1)
test_labels = torch.from_numpy(test_labels).long()

train_data = torch.from_numpy(train_data).float()
train_data = train_data.unsqueeze(1)
train_labels = torch.from_numpy(train_labels).long()
test_dataset = TensorDataset(test_data, test_labels)
test_num = len(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
predictions = []
with torch.no_grad():
    for data in test_loader:
        tdata, tlabels = data
        outputs = model(tdata.to(device))
        predict_y = torch.max(outputs, dim=1)[1]
        predictions.append(predict_y)

predictions = torch.cat(predictions, dim=0)
print(test_labels.shape)
print(predictions.shape)
conf_matrix = confusion_matrix(test_labels.cpu() , predictions.cpu())


# 计算每个类别的样本总数
class_counts = np.sum(conf_matrix, axis=1, keepdims=True)

# 归一化混淆矩阵为概率值
conf_matrix = conf_matrix / class_counts

# 绘制混淆矩阵图，并显示概率值
plt.figure(figsize=(7, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

# 在每个单元格中显示概率值
thresh = conf_matrix.max() / 2.0  # 阈值用于决定文本颜色
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, "{:.2f}".format(conf_matrix[i, j]), ha='center', va='center',
                 color='white' if conf_matrix[i, j] > thresh else 'black')

# plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(3)  # 假设有3个类别
plt.xticks(tick_marks, ['IR', 'OR', 'Healthy'])
plt.yticks(tick_marks, ['IR', 'OR', 'Healthy'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("log/plot_test/PhantomCNN_Large_matrix.png")
