import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random
import sys
from utils.set_seed import set_seed
from models.PhantomCNN import PhantomCNN
from models.WDCNN import WDCNN
from models.DRSN_CW import DRSN_CW
from models.MA1DCNN import MA1DCNN
from models.MIXCNN import MIXCNN
from utils.add_noise import batch_add_noise
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 超参数设置
# 随机数
set_seed(42)
# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 信噪比
snr_db = None
# 模型参数
model_params = {'channels':256, 'kernel_size':63, 'dilation':[1,2,5,1,2], 'phantomlayer_num':5,'num_classes':3}
# 模型权重
model_weights = "log/weights/PhantomCNN_Large_HIT_.pth"
# 数据路径
# data_path = "data/HIT/packaged/mix_displacement_data.npy"
data_path = "data/HIT/packaged/col0_data.npy"
# 数据标签路径
# labels_path = "data/HIT/packaged/mix_displacement_labels.npy"
labels_path = "data/HIT/packaged/col0_labels.npy"

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
        # predict_y = torch.max(outputs, dim=1)[1]
        predictions.append(outputs)

predictions = torch.cat(predictions, dim=0)
print(test_labels.shape)
print(predictions.shape)


y_true = test_labels.cpu()
y_scores = predictions.cpu()

# 将标签进行二值化
n_classes = 3
y_bin = label_binarize(y_true, classes=np.arange(n_classes))

# 计算每个类别的ROC曲线和AUC值
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制多类别ROC曲线
plt.figure()

colors = cycle(['lightblue', 'orange', 'lightgreen'])
class_ = ['IR', 'Normal','OR']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for multi-class')
plt.legend(loc="lower right")

plt.savefig("log/plot_test/roc_PhantomCNN_Large.png")
