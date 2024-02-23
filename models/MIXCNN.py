"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/2/20 下午3:20
@Email: 2909981736@qq.com
"""
import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile


class DepthwiseConv1D(nn.Module):
    def __init__(self, dim_in, kernel_size, padding="same", use_bias=False, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=dim_in, out_channels=dim_in, kernel_size=kernel_size, stride=1,
                              padding=padding, groups=dim_in, bias=use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class Mixconv(nn.Module):
    def __init__(self, channel=64, kersize=64, dim_in=64, dilation=1):
        super(Mixconv, self).__init__()
        self.depth_conv_1 = DepthwiseConv1D(dim_in=dim_in, kernel_size=kersize, padding="same", use_bias=False,
                                            dilation=dilation)
        self.act_2 = nn.ReLU()
        self.bn_2 = nn.BatchNorm1d(dim_in)
        self.point_conv_1 = nn.Conv1d(dim_in, channel, kernel_size=1, stride=1, padding="same")
        self.act_3 = nn.ReLU()
        self.bn_3 = nn.BatchNorm1d(channel)

    def forward(self, x):
        identity = x
        x = self.depth_conv_1(x)
        x = self.act_2(x)
        x = self.bn_2(x)
        x = torch.add(x, identity)
        x = self.point_conv_1(x)
        x = self.act_3(x)
        x = self.bn_3(x)
        return x


def first_layer_hook_fn(module, input, output):
    module.first_layer_feature_map = output


def mix_layer_1_hook_fn(module, input, output):
    module.mix_layer_1_feature_map = output


def mix_layer_2_hook_fn(module, input, output):
    module.mix_layer_2_feature_map = output


def mix_layer_3_hook_fn(module, input, output):
    module.mix_layer_3_feature_map = output


class MIXCNN(nn.Module):
    def __init__(self, dim_mid=128, conv1_ksize=32, stride=4, mixconv_ksize=64, num_classes=10, dilation=[1, 1, 1]):
        super(MIXCNN, self).__init__()
        # out = (in+2*p-k)/s+1
        self.conv_1 = nn.Conv1d(1, dim_mid, kernel_size=conv1_ksize, stride=stride)
        self.bn_1 = nn.BatchNorm1d(dim_mid)
        self.act_1 = nn.ReLU()
        self.mix_1 = Mixconv(dim_in=dim_mid, channel=dim_mid, kersize=mixconv_ksize, dilation=dilation[0])
        self.mix_2 = Mixconv(dim_in=dim_mid, channel=dim_mid, kersize=mixconv_ksize, dilation=dilation[1])
        self.mix_3 = Mixconv(dim_in=dim_mid, channel=dim_mid, kersize=mixconv_ksize, dilation=dilation[2])
        self.act_2 = nn.ReLU()
        self.bn_2 = nn.BatchNorm1d(dim_mid)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dim_mid, num_classes)

        # 注册钩子函数
        self.conv_1.register_forward_hook(first_layer_hook_fn)
        self.mix_1.register_forward_hook(mix_layer_1_hook_fn)
        self.mix_2.register_forward_hook(mix_layer_2_hook_fn)
        self.mix_3.register_forward_hook(mix_layer_3_hook_fn)

        # 用于保存中间特征图的变量
        self.first_layer_feature_map = None
        self.mix_layer_1_feature_map = None
        self.mix_layer_2_feature_map = None
        self.mix_layer_3_feature_map = None

    def forward(self, x):
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.bn_1(x)
        x = self.mix_1(x)
        x = self.mix_2(x)
        x = self.mix_3(x)
        x = self.act_2(x)
        x = self.bn_2(x)
        x = self.gap(x).squeeze()
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    input = torch.rand(1, 1, 1024).cuda()
    model1 = MIXCNN(dim_mid=64, conv1_ksize=16, mixconv_ksize=32, num_classes=10).cuda()
    model2 = MIXCNN(dim_mid=128, conv1_ksize=32, mixconv_ksize=64, num_classes=10).cuda()
    summary(model2, (1, 1024))
    output = model2(input)
    # output = torch.softmax(output, dim=1)
    # print(output)
    print(output.shape)
    flops, params = profile(model2, inputs=(input,))
    print("FLOPs=", str(flops) + '{}'.format("G"))
    print("Params=", str(params) + '{}'.format("M"))
