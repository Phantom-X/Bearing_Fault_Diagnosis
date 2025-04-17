"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/3/5 下午1:15
@Email: 2909981736@qq.com
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_heartmap_feature_map(feature_map, downsample_factor, cmap, show=True, savename=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # 计算降采样后的网格大小
    grid_size = (feature_map[0].shape[0] // downsample_factor, feature_map[0].shape[1] // downsample_factor)
    titles = ['(a) Phantom layer 4 PointwiseConv',
              '(b) Phantom layer 4 DilatationDepthwiseConv']
            #                 '(c) Phantom layer 2 PointwiseConv',
            #   '(d) Phantom layer 2 DilatationDepthwiseConv[dilated rate=2]',
            #   '(e) Phantom layer 3 PointwiseConv',
            #   '(f) Phantom layer 3 DilatationDepthwiseConv[dilated rate=5]',
            #   '(g) Phantom layer 4 PointwiseConv',
            #   '(h) Phantom layer 4 DilatationDepthwiseConv[dilated rate=1]'
    axs = axs.ravel()
    for i in range(len(axs)):
        # 使用均值池化进行降采样
        heatmap = feature_map[i].reshape(grid_size[0], downsample_factor, grid_size[1], downsample_factor).mean(
            axis=(1, 3))
        axs[i].imshow(heatmap, cmap='jet')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Spatial')
        axs[i].set_ylabel('Channel')
        axs[i].set_xticks(range(0, 256//downsample_factor, 50))
        axs[i].set_yticks(range(0, 64//downsample_factor, 10))
        # axs[i].axis('off')
        # axs[0].text(primary_heatmap.shape[1] // 2, 30, 'Spatial', ha='center', fontsize=12)
        # axs[0].text(-6, primary_heatmap.shape[0] // 2, 'Channel', va='center', rotation='vertical', fontsize=12)

    # 调整子图布局
    plt.tight_layout()

    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    plt.close()
