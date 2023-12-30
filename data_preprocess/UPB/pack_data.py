"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/29 下午2:50
@Email: 2909981736@qq.com
"""
import numpy as np
from glob import glob


class_dict = {
    'man-made_damage': {0: ['KI01', 'KI05', 'KI07'], 1: ['KA01', 'KA05', 'KA07'], 2: ['K002']},
    'naturally_damaged': {0: ['KI14', 'KI16', 'KI17', 'KI18', 'KI21'], 1: ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'],
                          2: ['K001']}
}


def pack_data(data_path, save_path, work='N15_M01_F10', length=1024, step_size=207, count=2000, seed=42):
    np.random.seed(seed)
    if work == -1:
        pass
    else:
        dataset_name = data_path.split('/')[-1]
        all_class_data = []
        for l in range(3):
            sample_count = count//len(class_dict[dataset_name][l])
            one_class_data = []
            for c in class_dict[dataset_name][l]:
                npy_files = glob(data_path+f'/{c}/{work}*.npy')
                all_sample_data = []
                file_count = sample_count//len(npy_files)
                for npy_file in npy_files:
                    data = np.load(npy_file)
                    window_data = np.array(
                        [data[i:i + length] for i in range(0, len(data) - length + 1, step_size)])
                    if file_count > len(window_data):
                        raise ValueError(f"所需数据长度超过数组长度 file_count({file_count}) > window_data({len(window_data)})")
                    sample_index = np.random.choice(np.arange(0, len(window_data)), size=file_count, replace=False)
                    sample_data = window_data[sample_index]
                    all_sample_data.extend(sample_data)
                one_class_data.extend(all_sample_data)
            all_class_data.extend(one_class_data)

        labels = np.arange(3)
        all_class_labels = np.repeat(labels, repeats=count)
        np.save(f"{save_path}/{dataset_name}_{work}_data.npy", all_class_data)
        np.save(f"{save_path}/{dataset_name}_{work}_labels.npy", all_class_labels)



if __name__ == '__main__':
    pack_data(data_path='../../data/UPB/naturally_damaged', save_path='../../data/UPB/packaged/',
              work='N15_M01_F10', length=2560,
              step_size=2560, count=1800, seed=42)
    print('ok')
