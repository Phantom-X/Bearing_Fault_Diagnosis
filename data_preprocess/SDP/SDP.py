"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/1/15 下午3:12
@Email: 2909981736@qq.com
"""
import os
import cv2 as cv
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


def SDP(x, savepath, g=0.69813172, time_lag=1, symmetry_number=6, imgdpi=224, isgray=False, color='blue', _color='blue',
        pensize=0.1, isline=True, show_coordinate_system=False, showimg=False):
    '''
    :param x: 一维数字信号（一维列表）
    :param savepath: SDP图像保存路径
    :param g: 角度放大倍数
    :param time_lag: 时间滞后数
    :param symmetry_number:对称平面个数，根据对称面旋转角度计算出
    :param imgdpi:图片大小
    :param isgray:是否灰度化
    :param color:顺时针图像颜色
    :param _color:逆时针图像颜色
    :param pensize:图像粗细
    :param isline:是否连线（折线图）
    :param show_coordinate_system:是否显示坐标系
    :param showimg:是否展示图片
    :return:
    '''
    l = len(x)
    xmax = max(x)
    xmin = min(x)

    fig = plt.figure(figsize=(1, 1), dpi=imgdpi)
    ax = plt.subplot(111, projection='polar')
    pp = []  # 数据关于对称轴顺时针方向的极坐标角度列表
    _pp = []  # 数据关于对称轴逆时针方向的极坐标角度列表
    rr = []  # 数据极坐标半径列表
    syaxisangle = (2 * np.pi) / symmetry_number  # 对称轴角度变换度数
    syaxisanglelist = []  # 对称轴角度列表
    for i in range(symmetry_number):
        syaxisanglelist.append(i * syaxisangle)
    for angle0 in syaxisanglelist:
        for n in range(l - time_lag):
            r = ((x[n] - xmin) / (xmax - xmin))  # 单个数据点极坐标半径
            rr.append(r)
            theta = angle0 - ((x[n + time_lag] - xmin) / (xmax - xmin)) * g  # 单个数据点关于对称轴顺时针方向的极坐标角度列表
            pp.append(theta)
            _theta = angle0 + ((x[n + time_lag] - xmin) / (xmax - xmin)) * g  # 单个数据点关于对称轴逆时针方向的极坐标角度列表
            _pp.append(_theta)
            # area = 100 * np.arange(1, 9, 1)  # 数据散点面积

    if isline:  # 折线图
        ax.plot(pp, rr, color=color, lw=pensize)
        ax.plot(_pp, rr, color=color, lw=pensize)
    else:  # 散点图
        ax.scatter(pp, rr, c=color, s=pensize, cmap='hsv')
        ax.scatter(_pp, rr, c=_color, s=pensize, cmap='hsv')
    if not show_coordinate_system:
        ax.spines['polar'].set_visible(False)
        ax.set_rgrids(np.arange(1.0, 1.0, 1.0))
        ax.set_thetagrids(np.arange(360.0, 360.0, 360.0))
    plt.savefig(savepath)
    if not isgray and showimg:
        plt.show()
    plt.close()
    if isgray:
        img = cv.imread(savepath + '.png', 0)

        cv.imwrite(savepath + '.png', img)
        if showimg:
            cv.imshow(savepath + '.png', img)
            cv.waitKey(0)


def TenSDP(datalist, savepath, g=0.69813172, time_lag=7, symmetry_number=6, imgdpi=680, isgray=False, color='blue',
           _color='blue',
           pensize=0.1,
           isline=True,
           show_coordinate_system=True, showimg=True):
    fig = plt.figure(figsize=(11, 5), dpi=imgdpi)
    axlist = [plt.subplot(251, projection='polar'),
              plt.subplot(252, projection='polar'),
              plt.subplot(253, projection='polar'),
              plt.subplot(254, projection='polar'),
              plt.subplot(255, projection='polar'),
              plt.subplot(256, projection='polar'),
              plt.subplot(257, projection='polar'),
              plt.subplot(258, projection='polar'),
              plt.subplot(259, projection='polar'),
              plt.subplot(2, 5, 10, projection='polar')]

    for di in range(len(datalist)):
        x = datalist[di]
        l = len(x)
        xmax = max(x)
        xmin = min(x)

        pp = []  # 数据关于对称轴顺时针方向的极坐标角度列表
        _pp = []  # 数据关于对称轴逆时针方向的极坐标角度列表
        rr = []  # 数据极坐标半径列表
        syaxisangle = (2 * np.pi) / symmetry_number  # 对称轴角度变换度数
        syaxisanglelist = []  # 对称轴角度列表
        for i in range(symmetry_number):
            syaxisanglelist.append(i * syaxisangle)
        for angle0 in syaxisanglelist:
            for n in range(l - time_lag):
                r = ((x[n] - xmin) / (xmax - xmin))  # 单个数据点极坐标半径
                rr.append(r)
                theta = angle0 - ((x[n + time_lag] - xmin) / (xmax - xmin)) * g  # 单个数据点关于对称轴顺时针方向的极坐标角度列表
                pp.append(theta)
                _theta = angle0 + ((x[n + time_lag] - xmin) / (xmax - xmin)) * g  # 单个数据点关于对称轴逆时针方向的极坐标角度列表
                _pp.append(_theta)
                # area = 100 * np.arange(1, 9, 1)  # 数据散点面积

        if isline:  # 折线图
            axlist[di].plot(pp, rr, color=color, lw=pensize)
            axlist[di].plot(_pp, rr, color=color, lw=pensize)
        else:  # 散点图
            axlist[di].scatter(pp, rr, c=color, s=pensize, cmap='hsv')
            axlist[di].scatter(_pp, rr, c=_color, s=pensize, cmap='hsv')
        if not show_coordinate_system:
            axlist[di].spines['polar'].set_visible(False)
            axlist[di].set_rgrids(np.arange(1.0, 1.0, 1.0))
            axlist[di].set_thetagrids(np.arange(360.0, 360.0, 360.0))
    plt.savefig(savepath)
    plt.show()
    plt.close()


if __name__ == '__main__':
    npydatafile = "../data/CRWU/packaged/12kDE_mix_data.npy"
    data = []
    x = np.load(npydatafile)
    for i in range(8, int(x.shape[0]), int(x.shape[0] / 10)):
        print(x[i, :].shape)
        data.append(x[i, :])

    savepath = f'../log/plot/12kDE_mix_data_10类别SDP图像2'
    TenSDP(data, savepath, g=0.69813172, time_lag=1, pensize=0.1, imgdpi=400)

    # data = readnpy(f'../save_data/npy_data/{npydatafiles[9]:}')
    # savepath = f'../save_data/testSDPimg/小波/SDP'
    # data_wavelet = wavelet_noising(data[245])
    # SDP(data_wavelet, savepath, g=0.69813172, time_lag=7, pointsize=0.1, imgdpi=680)
