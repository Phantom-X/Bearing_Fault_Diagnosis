"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/1/2 下午7:58
@Email: 2909981736@qq.com
"""
import torch
from torch import nn
from utils.set_seed import set_seed
from torchsummary import summary
from thop import profile

set_seed(42)


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
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, stride=1, padding='same', bias=False):
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
                                          padding=padding)
        self.cheap_conv = DepthwiseConv(in_channels=primary_channels, out_channels=cheap_channels,
                                        groups=cheap_channels, kernel_size=kernel_size, dilation=dilation,
                                        stride=stride, padding=padding)

    def forward(self, x):
        y = self.primary_conv(x)
        z = self.cheap_conv(y)
        z = torch.add(y, z)
        out = torch.cat([y, z], dim=1)
        return out


class PhantomLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=31, dilation=1, stride=1, padding='same'):
        super(PhantomLayer, self).__init__()
        self.phantom_conv1 = PhantomConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         dilation=dilation, stride=stride, padding=padding)
        self.p_relu1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.phantom_conv2 = PhantomConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         dilation=dilation, stride=stride, padding=padding)
        self.p_relu2 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x
        x = self.phantom_conv1(x)
        x = self.p_relu1(x)
        x = self.bn1(x)
        x = self.phantom_conv2(x)
        x = self.p_relu2(x)
        x = self.bn2(x)
        out = torch.add(x, identity)
        return out


class PhantomCNN(nn.Module):
    def __init__(self, channels=96, kernel_size=31, dilation=[1, 2, 5, 1, 2], num_classes=10):
        super(PhantomCNN, self).__init__()
        self.big_kernel_conv = nn.Conv1d(1, channels, kernel_size=16, stride=4, padding=6)  # out = (in+2*p-k)/s+1
        self.p_relu1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        self.phantomlayer1 = PhantomLayer(in_channels=channels, out_channels=channels, dilation=dilation[0],
                                          kernel_size=kernel_size)
        self.phantomlayer2 = PhantomLayer(in_channels=channels, out_channels=channels, dilation=dilation[1],
                                          kernel_size=kernel_size)
        self.phantomlayer3 = PhantomLayer(in_channels=channels, out_channels=channels, dilation=dilation[2],
                                          kernel_size=kernel_size)
        self.phantomlayer4 = PhantomLayer(in_channels=channels, out_channels=channels, dilation=dilation[3],
                                          kernel_size=kernel_size)
        self.phantomlayer5 = PhantomLayer(in_channels=channels, out_channels=channels, dilation=dilation[4],
                                          kernel_size=kernel_size)
        self.p_relu2 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(channels)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # 注册钩子函数
        # self.big_kernel_conv.register_forward_hook(first_layer_hook_fn)
        # self.phantomlayer1.register_forward_hook(phantom_layer1_hook_fn)
        # self.phantomlayer2.register_forward_hook(phantom_layer2_hook_fn)
        # self.phantomlayer3.register_forward_hook(phantom_layer3_hook_fn)
        # self.phantomlayer4.register_forward_hook(phantom_layer4_hook_fn)
        # self.phantomlayer5.register_forward_hook(phantom_layer5_hook_fn)
        # self.phantomlayer1.phantom_conv2.primary_conv.register_forward_hook(phantom_layer5_primary_conv_hook_fn)
        # self.phantomlayer1.phantom_conv2.cheap_conv.register_forward_hook(phantom_layer5_cheap_conv_hook_fn)
        # self.phantomlayer5.phantom_conv2.primary_conv.register_forward_hook(phantom_layer5_primary_conv_hook_fn)
        # self.phantomlayer5.phantom_conv2.cheap_conv.register_forward_hook(phantom_layer5_cheap_conv_hook_fn)

    def forward(self, x):
        x = self.big_kernel_conv(x)
        x = self.p_relu1(x)
        x = self.bn1(x)
        x = self.phantomlayer1(x)
        x = self.phantomlayer2(x)
        x = self.phantomlayer3(x)
        x = self.phantomlayer4(x)
        x = self.phantomlayer5(x)
        x = self.p_relu2(x)
        x = self.bn2(x)
        x = self.gap(x).view(x.shape[0], -1)
        x = self.fc(x)
        out = self.softmax(x)
        return out


if __name__ == '__main__':
    device = "cuda:0"
    x = torch.rand(1, 1, 1024).to(device)
    model = PhantomCNN().to(device)
    out = model(x)
    print(out.shape)
    summary(model, (1, 1024))
    flops, params = profile(model, inputs=(x,))
    print("FLOPs=", str(flops) + '{}'.format("G"))
    print("Params=", str(params) + '{}'.format("M"))
