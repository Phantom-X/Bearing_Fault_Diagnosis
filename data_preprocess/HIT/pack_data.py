"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/24 下午6:38
@Email: 2909981736@qq.com
"""
import os

import numpy as np
from glob import glob


def merge_sample(data_path, save_path, col, rotational_speed=None, length=1024, count=200, seed=42):
    np.random.seed(seed)
    if rotational_speed is None:
        npy_path = glob(data_path + "/*.npy")
        print(npy_path)
        for n, npy in enumerate(npy_path):
            data = np.load(npy)
            class_data = np.empty((0, 1024))
            for sample_data in data[:, col, :]:
                window_data = np.array(
                    [sample_data[i:i + length] for i in range(0, len(sample_data) - length + 1, length)])
                class_data = np.concatenate((class_data, window_data), axis=0)

            if count > len(class_data):
                raise ValueError(f"所需数据长度超过数组长度 count({count}) > class_data({len(class_data)})")
            sample_index = np.random.choice(np.arange(0, len(class_data)), size=count, replace=False)
            sample_data = class_data[sample_index]

            name_dict = {0: 'normal0', 1: 'normal1', 2: 'IR005', 3: 'IR010', 4: 'OR'}
            np.save(f"{save_path}/{name_dict[n]}.npy", sample_data)
    else:
        pass


def merge_class(data_path, save_path):
    npy_path = glob(data_path + "/*.npy")
    normal = []
    IR = []
    for npy in npy_path:
        data = np.load(npy)
        if 'normal' in npy:
            normal.append(data)
            os.remove(npy)
        elif 'IR' in npy:
            IR.append(data)
            os.remove(npy)
    normal = np.concatenate(normal, axis=0)
    IR = np.concatenate(IR, axis=0)

    np.save(f"{save_path}/normal.npy", normal)
    np.save(f"{save_path}/IR.npy", IR)


def pack_data(data_path, save_path, col, class_num=3, rotational_speed=None, length=1024, count=200, seed=42):
    np.random.seed(seed)
    if rotational_speed is None:
        npy_path = glob(data_path + "/*.npy")
        print(npy_path)
        all_sample_data = []
        for npy in npy_path:
            class_data = np.load(npy)
            if count > len(class_data):
                raise ValueError(f"所需数据长度超过数组长度 count({count}) > class_data({len(class_data)})")
            sample_index = np.random.choice(np.arange(0, len(class_data)), size=count, replace=False)
            sample_data = class_data[sample_index]
            all_sample_data.extend(sample_data)

        labels = np.arange(class_num)
        all_sample_labels = np.repeat(labels, repeats=count)
        np.save(f"{save_path}/col{col}_data.npy", all_sample_data)
        np.save(f"{save_path}/col{col}_labels.npy", all_sample_labels)
    else:
        pass


if __name__ == '__main__':
    col = 5
    merge_sample(data_path='../../data/raw_data/HIT', save_path='../../data/HIT', col=col,
                 length=1024, count=9000, seed=42)
    merge_class(data_path='../../data/HIT', save_path='../../data/HIT')
    pack_data(data_path='../../data/HIT', save_path='../../data/HIT/packaged', col=col,
              length=1024, count=1000, seed=42)
    print('ok')
