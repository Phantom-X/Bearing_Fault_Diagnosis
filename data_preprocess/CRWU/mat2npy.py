"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/17 下午3:31
@Email: 2909981736@qq.com
"""
import os
from matdata_ok import matdata
from scipy.io import loadmat
import numpy as np

for key in matdata.keys():
    os.mkdir(f'../../data/crwu/{key}')
    end = key[-2:]
    dataset = matdata[key]
    for class_ in dataset:
        class_dir = f'../../data/crwu/{key}/{str(class_["class"]) + "_" + class_["classname"]}'
        os.mkdir(class_dir)
        if 'OR' in class_["classname"]:
            for i, load in enumerate(class_['srcurl']):
                one_load_data = []
                for matfile in load:
                    mat = loadmat(matfile)
                    for mat_key in mat.keys():
                        if end in mat_key:
                            one_load_data.append(mat[mat_key].ravel())
                one_load_data_numpy = np.concatenate(one_load_data)
                np.save(f'{class_dir}/{str(i)+".npy"}', one_load_data_numpy)
        else:
            for i, matfile in enumerate(class_['srcurl']):
                mat = loadmat(matfile)
                for mat_key in mat.keys():
                    if end in mat_key:
                        one_load_data_numpy = mat[mat_key].ravel()
                        np.save(f'{class_dir}/{str(i)+".npy"}', one_load_data_numpy)
                        break
        print(class_dir, 'OK')