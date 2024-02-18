"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/25 下午7:29
@Email: 2909981736@qq.com
"""
import numpy as np
from glob import glob


def pack_data(data_path, rotating_speed, save_path, length=1024, step_size=300, count=200, seed=42):
    np.random.seed(seed)
    if rotating_speed == -1:
        classes = ['ib', 'n', 'ob', 'tb']
        class_num = 4
        all_sample_data = []
        for class_ in classes:
            class_npys = glob(data_path + f"/{class_}*.npy")
            class_all_data = []
            for class_npy in class_npys:
                data = np.load(class_npy)
                window_data = np.array(
                    [data[i:i + length] for i in range(0, len(data) - length + 1, step_size)])
                class_all_data.append(window_data)
            class_data = np.concatenate(class_all_data, axis=0)
            if count > len(class_data):
                raise ValueError(f"所需数据长度超过数组长度 count({count}) > class_data({len(class_data)})")
            sample_index = np.random.choice(np.arange(0, len(class_data)), size=count, replace=False)
            sample_data = class_data[sample_index]
            all_sample_data.extend(sample_data)
        labels = np.arange(class_num)
        all_sample_labels = np.repeat(labels, repeats=count)
        np.save(f"{save_path}/jnu_r_mix_data.npy", all_sample_data)
        np.save(f"{save_path}/jnu_r_mix_labels.npy", all_sample_labels)
    else:
        npyfile = glob(data_path + f"/*{rotating_speed}*.npy")
        print(npyfile)
        class_num = 4
        all_sample_data = []
        for npy in npyfile:
            class_data = np.load(npy)
            window_data = np.array(
                [class_data[i:i + length] for i in range(0, len(class_data) - length + 1, step_size)])
            if count > len(window_data):
                raise ValueError(f"所需数据长度超过数组长度 count({count}) > window_data({len(window_data)})")
            sample_index = np.random.choice(np.arange(0, len(window_data)), size=count, replace=False)
            sample_data = window_data[sample_index]
            all_sample_data.extend(sample_data)
        labels = np.arange(class_num)
        all_sample_labels = np.repeat(labels, repeats=count)
        np.save(f"{save_path}/jnu_r_{rotating_speed}_data.npy", all_sample_data)
        np.save(f"{save_path}/jnu_r_{rotating_speed}_labels.npy", all_sample_labels)


if __name__ == '__main__':
    pack_data(data_path='../../data/JNU', rotating_speed=1000, save_path='../../data/JNU/packaged', length=1024,
              step_size=1024, count=480, seed=42)
    print("ok")
