"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/3/6 下午1:41
@Email: 2909981736@qq.com
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft

def plot_feature_signal_diagram(feature_map, sampling_rate=12000, show=True, savename=None):
    fig, axs = plt.subplots(6, 8, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)  # 调整子图之间的垂直间距
    axs = axs.ravel()
    index = np.arange(256)
    for i in range(len(axs)):
        axs[i].plot(index, feature_map[0][i], color="blue")
        axs[i].set_title(f"({i + 1})", pad=-150)  # 设置标题的垂直偏移量
        axs[i].axis('on')  # 显示坐标框
        axs[i].tick_params(axis='both', which='both', bottom=False, top=False,
                           labelbottom=False, left=False, right=False, labelleft=False)  # 不显示坐标值和刻度
    if savename is not None:
        plt.savefig(savename.split(".png")[0] + "_primary_conv_feature_signal_diagram.png")
    if show:
        plt.show()
    plt.close()

    fig, axs = plt.subplots(6, 8, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)  # 调整子图之间的垂直间距
    axs = axs.ravel()
    for i in range(len(axs)):
        ft = fft(feature_map[0][i])  # 需要注意 只能对一个通道的数据进行操作
        magnitude = np.absolute(ft)  # 取相似度
        magnitude = magnitude[0:int(len(magnitude) / 2) + 1]
        f = np.linspace(0, sampling_rate, len(magnitude))
        axs[i].plot(f, magnitude, color="blue")
        # spectrum = np.fft.fft(feature_map[0][i])
        # frequencies = np.fft.fftfreq(len(feature_map[0][i]), d=1/sampling_rate)
        # axs[i].plot(frequencies, np.abs(spectrum), color="blue")
        axs[i].set_title(f"({i + 1})", pad=-150)  # 设置标题的垂直偏移量
        axs[i].axis('on')  # 显示坐标框
        axs[i].tick_params(axis='both', which='both', bottom=False, top=False,
                           labelbottom=False, left=False, right=False, labelleft=False)  # 不显示坐标值和刻度
    if savename is not None:
        plt.savefig(savename.split(".png")[0] + "_primary_conv_feature_Spectrogram.png")
    if show:
        plt.show()
    plt.close()

    fig, axs = plt.subplots(6, 8, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)  # 调整子图之间的垂直间距
    axs = axs.ravel()
    for i in range(len(axs)):
        axs[i].plot(index, feature_map[1][i], color="red")
        axs[i].set_title(f"({i + 1})", pad=-150)  # 设置标题的垂直偏移量
        axs[i].axis('on')  # 显示坐标框
        axs[i].tick_params(axis='both', which='both', bottom=False, top=False,
                           labelbottom=False, left=False, right=False, labelleft=False)  # 不显示坐标值和刻度
    if savename is not None:
        plt.savefig(savename.split(".png")[0] + "_cheap_conv_feature_signal_diagram.png")
    if show:
        plt.show()
    plt.close()

    fig, axs = plt.subplots(6, 8, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)  # 调整子图之间的垂直间距
    axs = axs.ravel()
    for i in range(len(axs)):
        ft = fft(feature_map[1][i])  # 需要注意 只能对一个通道的数据进行操作
        magnitude = np.absolute(ft)  # 取相似度
        magnitude = magnitude[0:int(len(magnitude) / 2) + 1]
        f = np.linspace(0, sampling_rate, len(magnitude))
        axs[i].plot(f, magnitude, color="red")
        # spectrum = np.fft.fft()
        # frequencies = np.fft.fftfreq(len(feature_map[1][i]), d=1/sampling_rate)
        # axs[i].plot(frequencies, np.abs(spectrum), color="red")
        axs[i].set_title(f"({i + 1})", pad=-150)  # 设置标题的垂直偏移量
        axs[i].axis('on')  # 显示坐标框
        axs[i].tick_params(axis='both', which='both', bottom=False, top=False,
                           labelbottom=False, left=False, right=False, labelleft=False)  # 不显示坐标值和刻度
    if savename is not None:
        plt.savefig(savename.split(".png")[0] + "_cheap_conv_feature_Spectrogram.png")
    if show:
        plt.show()
    plt.close()
