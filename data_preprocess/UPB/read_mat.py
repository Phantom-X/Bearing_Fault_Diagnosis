"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/28 下午7:53
@Email: 2909981736@qq.com
"""
from scipy.io import loadmat


mat = loadmat('../../data/raw_Data/UPB/be_use/naturally_damaged/KA22/N15_M01_F10_KA22_6.mat')

print(mat)
# print(mat['N15_M01_F10_KA22_6'][0][0][2][0][6][2].shape)
# print(mat['N15_M01_F10_KA22_6'][0][0][2][0][6][2])
