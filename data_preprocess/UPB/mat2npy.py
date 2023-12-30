"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/29 下午2:58
@Email: 2909981736@qq.com
"""
import os

import numpy as np
from scipy.io import loadmat
from glob import glob


# ['N15_M01_F10_K001_1'][0][0][2][0][6][2]

def mat2npy(data_path, save_path, work):
    bearings = glob(data_path + "/*")
    for bearing in bearings:
        new_bearing_path = save_path + f'/{bearing.split("/")[-1]}'
        os.mkdir(new_bearing_path)
        mat_files = glob(bearing + f'/{work}*.mat')
        for mat_file in mat_files:
            try:
                mat = loadmat(mat_file)
                mat_name = mat_file.split('/')[-1].split('.')[0]
                vibration_signal = mat[mat_name][0][0][2][0][6][2].flatten()
                print(vibration_signal.shape)
                np.save(new_bearing_path + f'/{mat_name}.npy', vibration_signal)
            except:
                print(mat_file)


if __name__ == '__main__':
    mat2npy(data_path='../../data/raw_Data/UPB/be_use/naturally_damaged', save_path='../../data/UPB/naturally_damaged',
            work='N15_M01_F10')
