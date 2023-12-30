"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/16 下午5:10
@Email: 2909981736@qq.com
"""


import numpy as np

a = np.load('data/UPB/packaged/naturally_damaged_N15_M01_F10_data.npy')
b = np.load('data/UPB/packaged/naturally_damaged_N15_M01_F10_labels.npy')

print(a.shape)
print(b.shape)
print(b)

