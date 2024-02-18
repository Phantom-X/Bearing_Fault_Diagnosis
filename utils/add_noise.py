"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/30 下午7:39
@Email: 2909981736@qq.com
"""
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

np.random.seed(42)


def add_gaussian_noise(signal, snr_db):
    # 计算信号功率
    signal_power = np.sum(signal ** 2) / len(signal)
    # 计算噪音功率
    snr = 10 ** (snr_db / 10)  # 将信噪比（单位dB）转换为线性信噪比
    noise_power = signal_power / snr

    # 生成高斯噪音
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # 添加噪音到信号
    noisy_signal = signal + noise

    return noisy_signal


def plot_noise_fig(signal, snr_db, save_path=None):
    noisy_signal = add_gaussian_noise(signal, snr_db)
    x = np.arange(len(signal))
    plt.subplot(2, 1, 1)
    plt.plot(x, signal)
    plt.subplot(2, 1, 2)
    plt.plot(x, noisy_signal)
    plt.title('Noisy Signal (SNR = {}dB)'.format(snr_db))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def batch_add_noise(data, snr_db):
    if snr_db is not None:
        partial_add_gaussian_noise = partial(add_gaussian_noise, snr_db=snr_db)
        noisy_signal = list(map(partial_add_gaussian_noise, data))
        noisy_signal = np.array(noisy_signal)
        return noisy_signal
    else:
        return data


if __name__ == '__main__':
    data = np.load("../data/cwru_data.npy")
    print(data.shape)
    batch_add_noise(data, snr_db=-2)
