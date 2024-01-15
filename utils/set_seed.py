"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/1/15 下午3:04
@Email: 2909981736@qq.com
"""
import torch
import random
import numpy as np
import os


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
