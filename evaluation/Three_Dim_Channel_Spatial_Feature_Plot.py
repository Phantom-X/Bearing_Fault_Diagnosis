"""
@project:MIXCNN
@Author: Phantom
@Time:2023/10/31 下午1:41
@Email: 2909981736@qq.com
"""
import matplotlib.pyplot as plt
import numpy as np


# 3D Channel Spatial Feature Plot
# 3D 通道空间特征图
def plot_3dcsfp(feature_map, cmap, show=True, savename=None):
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(232, projection='3d')
    ax3 = fig.add_subplot(233, projection='3d')
    ax4 = fig.add_subplot(234, projection='3d')
    ax5 = fig.add_subplot(235, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')
    # ax7 = fig.add_subplot(247, projection='3d')
    # ax8 = fig.add_subplot(248, projection='3d')

    # Set titles and labels for each subplot
    ax1.set_title('(a) Phantom layer 1')
    ax1.set_xlabel('Spatial')
    ax1.set_ylabel('Channel')
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel('Feature', rotation=90)

    ax2.set_title('(b) Phantom layer 2')
    ax2.set_xlabel('Spatial')
    ax2.set_ylabel('Channel')
    ax2.zaxis.set_rotate_label(False)
    ax2.set_zlabel('Feature', rotation=90)

    ax3.set_title('(c) Phantom layer 3')
    ax3.set_xlabel('Spatial')
    ax3.set_ylabel('Channel')
    ax3.zaxis.set_rotate_label(False)
    ax3.set_zlabel('Feature', rotation=90)

    ax4.set_title('(d) Phantom layer 4')
    ax4.set_xlabel('Spatial')
    ax4.set_ylabel('Channel')
    ax4.zaxis.set_rotate_label(False)
    ax4.set_zlabel('Feature', rotation=90)

    ax5.set_title('(e) Phantom layer 4 PointwiseConv')
    ax5.set_xlabel('Spatial')
    ax5.set_ylabel('Channel')
    ax5.zaxis.set_rotate_label(False)
    ax5.set_zlabel('Feature', rotation=90)

    ax6.set_title('(f) Phantom layer 4 DilatationDepthwiseConv')
    ax6.set_xlabel('Spatial')
    ax6.set_ylabel('Channel')
    ax6.zaxis.set_rotate_label(False)
    ax6.set_zlabel('Feature', rotation=90)

    # ax7.set_title('(g) Phantom layer 5 primary conv')
    # ax7.set_xlabel('Spatial')
    # ax7.set_ylabel('Channel')
    # ax7.zaxis.set_rotate_label(False)
    # ax7.set_zlabel('Feature', rotation=90)

    # ax8.set_title('(h) Phantom layer 5 cheap conv')
    # ax8.set_xlabel('Spatial')
    # ax8.set_ylabel('Channel')
    # ax8.zaxis.set_rotate_label(False)
    # ax8.set_zlabel('Feature', rotation=90)

    # jet hsv rainbow viridis
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):  # , ax7, ax8
        Y = np.arange(0, feature_map[i].shape[1], 1)
        X = np.arange(0, feature_map[i].shape[2], 1)
        X, Y = np.meshgrid(X, Y)
        Z = feature_map[i]
        Z = Z[0]  # plot_surface
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap(cmap), linewidth=1)  # antialiased=False
        # 设置颜色映射范围
        # surf.set_clim(-1, 1)
        ax.set_xlim(feature_map[i].shape[2], 0)
        ax.set_ylim(feature_map[i].shape[1], 0)
        ax.set_zlim(np.min(Z), np.max(Z))
        ax.view_init(elev=10, azim=35)

        # # 添加颜色栏
        # fig.colorbar(surf)
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    plt.close()

