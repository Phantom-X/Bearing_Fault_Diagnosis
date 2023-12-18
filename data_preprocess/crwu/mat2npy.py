"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/17 下午3:31
@Email: 2909981736@qq.com
"""
from matdata_ok import matdata
from scipy.io import loadmat

for mat in matdata:
    src_data = loadmat(mat['srcurl'])
    print(src_data['X097_DE_time'])

    break



