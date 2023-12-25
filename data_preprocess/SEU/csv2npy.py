"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/20 下午6:51
@Email: 2909981736@qq.com
"""
import pandas as pd
import numpy as np
from glob import glob
import csv
import os

def detect_delimiter(filename):
    with open(filename, 'r', newline='') as file:
        dialect = csv.Sniffer().sniff(file.read(1024))
        return dialect.delimiter


data_root_path = ['../../data/raw_data/SEU/bearingset', '../../data/raw_data/SEU/gearset']

for data_root in data_root_path:
    save_path = f'../../data/SEU/{data_root.split("/")[-1]}'
    os.mkdir(save_path)
    all_csv = glob(data_root + '/*.csv')
    for one_csv in all_csv:
        print(one_csv)
        delimiter = detect_delimiter(one_csv)
        data = pd.read_csv(one_csv, skiprows=15, sep=delimiter, usecols=[0, 1, 2, 3, 4, 5, 6, 7])
        data_array = np.empty((len(data.columns), len(data)))
        for i, (col_name, col_data) in enumerate(data.iteritems()):
            data_array[i, :] = col_data.values
        np.save(f"{save_path}/{one_csv.split('/')[-1].split('.')[0]}.npy", data_array)