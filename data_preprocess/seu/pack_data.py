"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/18 下午6:06
@Email: 2909981736@qq.com
"""
import numpy as np
from glob import glob


# col=1, 5
def pack_data(data_path, load, save_path, col, length=1024, count=200, seed=42):
    np.random.seed(seed)
    if load == -1:
        if isinstance(data_path, str):
            if not isinstance(col, int):
                raise ValueError(f'col（{col}）在该状态下必须是整数0-7中的一个')
            class_num = 5
            classes = glob(data_path + f'/*.npy')
            all_sample_data = []
            for l in range(0, len(classes), 2):
                class_data = []
                for one_load in [classes[l], classes[l+1]]:
                    one_load_data = np.load(one_load)[col]
                    window_data = np.array(
                        [one_load_data[i:i + length] for i in range(0, len(one_load_data) - length + 1, length)])
                    class_data.append(window_data)
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

        elif isinstance(data_path, list):
            if not isinstance(col, list):
                raise ValueError(f'col（{col}）在该状态下必须是一个长为2的列表，其中每个数字都在0-8之间，例如 col = [1, 5]')
            class_num = 10
            all_sample_data = []
            for c, path in enumerate(data_path):
                classes = glob(path + f'/*.npy')
                for l in range(0, len(classes), 2):
                    class_data = []
                    for one_load in [classes[l], classes[l + 1]]:
                        one_load_data = np.load(one_load)[col[c]]
                        window_data = np.array(
                            [one_load_data[i:i + length] for i in range(0, len(one_load_data) - length + 1, length)])
                        class_data.append(window_data)
                    window_data = np.concatenate(class_data)
                    if count > len(window_data):
                        raise ValueError(f"所需数据长度超过数组长度 count({count}) > window_data({len(window_data)})")
                    sample_index = np.random.choice(np.arange(0, len(window_data)), size=count, replace=False)
                    sample_data = window_data[sample_index]
                    all_sample_data.extend(sample_data)
            labels = np.arange(class_num)
            all_sample_labels = np.repeat(labels, repeats=count)
            np.save(f"{save_path}/{data_path[0].split('/')[-1]}_{data_path[1].split('/')[-1]}_mix_data.npy",
                    all_sample_data)
            np.save(f"{save_path}/{data_path[0].split('/')[-1]}_{data_path[1].split('/')[-1]}_mix_labels.npy",
                    all_sample_labels)

    else:
        if load != '20_0' and load != '30_2':
            raise ValueError(f'load（{load}）必须是字符串"20_0"或者"30_2"或者整数-1')
        if isinstance(data_path, str):
            if not isinstance(col, int):
                raise ValueError(f'col（{col}）在该状态下必须是整数0-7中的一个')
            classes = glob(data_path + f'/*{load}*.npy')
            class_num = len(classes)
            all_sample_data = []
            for class_ in classes:
                class_data = np.load(class_)[col]
                window_data = np.array(
                    [class_data[i:i + length] for i in range(0, len(class_data) - length + 1, length)])
                if count > len(window_data):
                    raise ValueError(f"所需数据长度超过数组长度 count({count}) > window_data({len(window_data)})")
                sample_index = np.random.choice(np.arange(0, len(window_data)), size=count, replace=False)
                sample_data = window_data[sample_index]
                all_sample_data.extend(sample_data)

            labels = np.arange(class_num)
            all_sample_labels = np.repeat(labels, repeats=count)
            np.save(f"{save_path}/{data_path.split('/')[-1]}_{load}_data.npy", all_sample_data)
            np.save(f"{save_path}/{data_path.split('/')[-1]}_{load}_labels.npy", all_sample_labels)

        elif isinstance(data_path, list):
            if not isinstance(col, list):
                raise ValueError(f'col（{col}）在该状态下必须是一个长为2的列表，其中每个数字都在0-8之间，例如 col = [1, 5]')
            class_num = 10
            all_sample_data = []
            for c, path in enumerate(data_path):
                classes = glob(path + f'/*{load}*.npy')
                for class_ in classes:
                    class_data = np.load(class_)[col[c]]
                    window_data = np.array(
                        [class_data[i:i + length] for i in range(0, len(class_data) - length + 1, length)])
                    if count > len(window_data):
                        raise ValueError(f"所需数据长度超过数组长度 count({count}) > window_data({len(window_data)})")
                    sample_index = np.random.choice(np.arange(0, len(window_data)), size=count, replace=False)
                    sample_data = window_data[sample_index]
                    all_sample_data.extend(sample_data)

            labels = np.arange(class_num)
            all_sample_labels = np.repeat(labels, repeats=count)
            np.save(f"{save_path}/{data_path[0].split('/')[-1]}_{data_path[1].split('/')[-1]}_{load}_data.npy",
                    all_sample_data)
            np.save(f"{save_path}/{data_path[0].split('/')[-1]}_{data_path[1].split('/')[-1]}_{load}_labels.npy",
                    all_sample_labels)
        else:
            raise ValueError(f'data_path（{data_path}）必须是字符串str或列表list')


if __name__ == '__main__':
    pack_data(data_path='../../data/seu/bearingset', load='20_0', save_path='../../data/seu/packaged', col=1, length=1024, count=980, seed=42)
    pack_data(data_path='../../data/seu/bearingset', load='30_2', save_path='../../data/seu/packaged', col=1, length=1024, count=980, seed=42)
    pack_data(data_path='../../data/seu/bearingset', load=-1, save_path='../../data/seu/packaged', col=1, length=1024, count=980, seed=42)
    pack_data(data_path=['../../data/seu/bearingset', '../../data/seu/gearset'], load='20_0', save_path='../../data/seu/packaged', col=[1, 5], length=1024, count=980, seed=42)
    pack_data(data_path=['../../data/seu/bearingset', '../../data/seu/gearset'], load='30_2', save_path='../../data/seu/packaged', col=[1, 5], length=1024, count=980, seed=42)
    pack_data(data_path=['../../data/seu/bearingset', '../../data/seu/gearset'], load=-1, save_path='../../data/seu/packaged', col=[1, 5], length=1024, count=980, seed=42)
    print('ok')
