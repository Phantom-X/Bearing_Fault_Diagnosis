"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/16 下午5:10
@Email: 2909981736@qq.com
"""


import numpy as np

a = np.load('data/CRWU/packaged/12kDE_1_data.npy')
b = np.load('data/CRWU/packaged/12kDE_1_labels.npy')

print(a.shape)
print(b.shape)
