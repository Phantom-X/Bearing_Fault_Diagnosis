"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/18 下午6:06
@Email: 2909981736@qq.com
"""
import numpy as np
from glob import glob


def pack_data(data_path, load, save_path, length=1024, step_size=300, count=200, seed=42):
    """
    :param data_path:
    :param load:
    :param save_path:
    :param length:
    :param step_size:
    :param count:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    classes = glob(data_path + "/*")
    class_num = len(classes)
    if load == -1:
        all_sample_data = []
        for class_ in classes:
            class_data = []
            all_load = glob(class_ + '/*.npy')
            for one_load in all_load:
                one_load_data = np.load(one_load)
                for sub in range(len(one_load_data)):
                    class_data.append([one_load_data[sub][i:i + length] for i in
                                       range(0, len(one_load_data[sub]) - length + 1, step_size)])
            window_data = np.concatenate(class_data)
            if count > len(window_data):
                raise ValueError(f"所需数据长度超过数组长度 count({count}) > window_data({len(window_data)})")
            sample_index = np.random.choice(np.arange(0, len(window_data)), size=count, replace=False)
            sample_data = window_data[sample_index]
            all_sample_data.extend(sample_data)

        labels = np.arange(class_num)
        all_sample_labels = np.repeat(labels, repeats=count)
        np.save(f"{save_path}/{data_path.split('/')[-1]}_mix_data.npy", all_sample_data)
        np.save(f"{save_path}/{data_path.split('/')[-1]}_mix_labels.npy", all_sample_labels)

    else:
        all_sample_data = []
        for class_ in classes:
            class_data = np.load(class_ + "/" + str(load) + '.npy')
            all_window_data = []
            for sub in range(len(class_data)):
                all_window_data.append(
                    [class_data[sub][i:i + length] for i in range(0, len(class_data[sub]) - length + 1, step_size)])
            window_data = np.concatenate(all_window_data)
            if count > len(window_data):
                raise ValueError(f"所需数据长度超过数组长度 count({count}) > window_data({len(window_data)})")
            sample_index = np.random.choice(np.arange(0, len(window_data)), size=count, replace=False)
            sample_data = window_data[sample_index]
            all_sample_data.extend(sample_data)

        labels = np.arange(class_num)
        all_sample_labels = np.repeat(labels, repeats=count)
        np.save(f"{save_path}/{data_path.split('/')[-1]}_{load}_data.npy", all_sample_data)
        np.save(f"{save_path}/{data_path.split('/')[-1]}_{load}_labels.npy", all_sample_labels)


if __name__ == '__main__':
    pack_data(data_path='../../data/CWRU/12kDE', load=-1, save_path='../../data/CWRU/packaged', length=1024,
              step_size=207, count=2000, seed=42)
    print('ok')
