"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/3/6 上午9:24
@Email: 2909981736@qq.com
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


def plot_tsne_map(feature_map, labels, camp='rainbow', n_components=2, verbose=1, perplexity=100, n_iter=1000,
                  show=True, savename=None):
    tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    titles = ['(1) Phantom layer 1',
              '(2) Phantom layer 2',
              '(3) Phantom layer 3',
              '(4) Phantom layer 4']
    axs = axs.ravel()
    unique_labels = list(set(labels))  # 获取唯一的类别
    color_map = plt.cm.get_cmap(camp, len(unique_labels))  # 从 colormap 中获取颜色映射

    # 创建自定义图例
    custom_legend = []
    for label in unique_labels:
        custom_legend.append(axs[0].scatter([], [], c=color_map(label), label=label))

    for i in range(len(axs)):
        tsne_results = tsne.fit_transform(feature_map[i])
        scatter = axs[i].scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, s=5, cmap=camp)
        axs[i].set_title(titles[i])
        axs[i].grid(False)  # 关闭网格显示

    # 在大图上添加自定义图例
    fig.legend(handles=custom_legend, loc='center right', title='Label')
    if savename:
        plt.savefig(savename)
    if show:
        plt.show()


# def plot_tsne_map(feature_map, labels, camp='rainbow', n_components=2, verbose=1, perplexity=100, n_iter=1000,
#                   show=True, savename=None):
#     tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
#     fig, axs = plt.subplots(2, 2, figsize=(12, 12))
#     titles = ['(1) First layer t-sne cluster diagram',
#               '(2) MIX layer 1 t-sne cluster diagram',
#               '(3) MIX layer 2 t-sne cluster diagram',
#               '(4) MIX layer 3 t-sne cluster diagram']
#     axs = axs.ravel()
#     unique_labels = list(set(labels))  # 获取唯一的类别
#     color_map = plt.cm.get_cmap(camp, len(unique_labels))  # 从 colormap 中获取颜色映射
#
#     # 创建自定义图例
#     custom_legend = []
#     for label in unique_labels:
#         custom_legend.append(axs[0].scatter([], [], c=color_map(label), label=label))
#
#     for i in range(len(axs)):
#         tsne_results = tsne.fit_transform(feature_map[i])
#         scatter = axs[i].scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, s=5, cmap=camp)
#         axs[i].set_title(titles[i])
#
#     # 在大图上添加自定义图例
#     fig.legend(handles=custom_legend, loc='center right', title='Label')
#     if savename:
#         plt.savefig(savename)
#     if show:
#         plt.show()
