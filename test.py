"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/16 下午5:10
@Email: 2909981736@qq.com
"""

import pandas as pd
import numpy as np

# 创建一个示例DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)

# 初始化一个空的二维数组
result_array = np.empty((len(df), len(df.columns)))

# 遍历DataFrame的每一列，并将其存储到二维数组中
for i, (colname, coldata) in enumerate(df.iteritems()):
    result_array[:, i] = coldata.values

# 打印结果
print(result_array)

