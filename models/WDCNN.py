"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/2/18 下午4:07
@Email: 2909981736@qq.com
"""
import torch
from torch import nn


def cbrm_module(in_channel, out_channel, kernel_size, padding, stride):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )


class WDCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(WDCNN, self).__init__()
        self.cbrm1 = cbrm_module(in_channel=1, out_channel=16, kernel_size=64, padding=24, stride=16)
        self.cbrm2 = cbrm_module(in_channel=16, out_channel=32, kernel_size=3, padding='same', stride=1)
        self.cbrm3 = cbrm_module(in_channel=32, out_channel=64, kernel_size=3, padding='same', stride=1)
        self.cbrm4 = cbrm_module(in_channel=64, out_channel=64, kernel_size=3, padding='same', stride=1)
        self.cbrm5 = cbrm_module(in_channel=64, out_channel=64, kernel_size=3, padding='same', stride=1)
        self.fc = nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cbrm1(x)
        x = self.cbrm2(x)
        x = self.cbrm3(x)
        x = self.cbrm4(x)
        x = self.cbrm5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.softmax(x)
        return out


if __name__ == '__main__':
    wdcnn = WDCNN(num_classes=10)
    x = torch.randn((32, 1, 1024))
    out = wdcnn(x)
    print(out.shape)
