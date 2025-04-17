"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/1/2 下午7:58
@Email: 2909981736@qq.com
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import math
import torch
from torch import nn
from utils.set_seed import set_seed
from torchinfo import summary
from thop import profile

set_seed(42)


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1)
        return out * x


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups, kernel_size=31, dilation=1, stride=1, padding='same',
                 bias=False):
        super(DepthwiseConv, self).__init__()
        self.dw_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, groups=groups,
                                 kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, bias=bias)
        self.p_relu = nn.PReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.p_relu(x)
        x = self.bn(x)
        return x


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, stride=1, padding='same', bias=False):
        super(PointwiseConv, self).__init__()
        self.pw_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 dilation=dilation, stride=stride, padding=padding, bias=bias)
        self.p_relu = nn.PReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.pw_conv(x)
        x = self.p_relu(x)
        x = self.bn(x)
        return x


class PhantomConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=31, dilation=1, stride=1, padding='same'):
        super(PhantomConv, self).__init__()
        primary_channels = out_channels // 2
        cheap_channels = out_channels - primary_channels
        self.primary_conv = PointwiseConv(in_channels=in_channels, out_channels=primary_channels, stride=stride,
                                          padding=padding, bias=True)
        self.cheap_conv = DepthwiseConv(in_channels=primary_channels, out_channels=cheap_channels,
                                        groups=cheap_channels, kernel_size=kernel_size, dilation=dilation,
                                        stride=stride, padding=padding)

    def forward(self, x):
        y = self.primary_conv(x)
        z = self.cheap_conv(y)
        out = torch.cat([y, z], dim=1)
        return out


class PhantomLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=31, dilation=1, stride=1, padding='same'):
        super(PhantomLayer, self).__init__()
        self.phantom_conv1 = PhantomConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         dilation=dilation, stride=stride, padding=padding)
        self.p_relu1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.phantom_conv2 = PhantomConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        #                                  dilation=dilation, stride=stride, padding=padding)
        # self.p_relu2 = nn.PReLU()
        # self.bn2 = nn.BatchNorm1d(out_channels)
        self.eca = ECA(in_channel=96)

    def forward(self, x):
        identity = x
        identity = self.eca(identity)
        x = self.phantom_conv1(x)
        x = self.p_relu1(x)
        x = self.bn1(x)
        # x = self.phantom_conv2(x)
        # x = self.p_relu2(x)
        # x = self.bn2(x)
        out = torch.add(x, identity)
        return out


class PhantomCNN(nn.Module):
    def __init__(self, channels=128, kernel_size=63, dilation=[1, 2, 5, 1], num_classes=10, phantomlayer_num=4):
        super(PhantomCNN, self).__init__()
        self.big_kernel_conv = nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6)  # out = (in+2*p-k)/s+1
        self.p_relu1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        phantomlayers = []
        assert len(dilation)==phantomlayer_num
        for i in range(phantomlayer_num):
            phantomlayers.append(PhantomLayer(in_channels=channels, out_channels=channels, dilation=dilation[i],
                                          kernel_size=kernel_size))
        self.phantomlayers = nn.Sequential(*phantomlayers)
        self.p_relu2 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(channels)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.big_kernel_conv(x)
        x = self.p_relu1(x)
        x = self.bn1(x)
        x = self.phantomlayers(x)
        x = self.p_relu2(x)
        x = self.bn2(x)
        x = self.gap(x).view(x.shape[0], -1)
        x = self.fc(x)
        out = self.softmax(x)
        return out


if __name__ == '__main__':
    device = "cuda:0"
    x = torch.rand(1, 1, 1024).to(device)
    model = PhantomCNN(channels=128,kernel_size=63,num_classes=10,dilation=[1,2,5,1], phantomlayer_num=4).to(device)
    # model = PhantomCNN(channels=256,kernel_size=63,num_classes=10,dilation=[1,2,5,1,2], phantomlayer_num=5).to(device)
    out = model(x)
    print(out.shape)
    summary(model, (1, 1, 1024))
    flops, params = profile(model, inputs=(x,))
    print("FLOPs=", str(flops) + '{}'.format(""))
    print("Params=", str(params) + '{}'.format(""))
