"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/2/18 下午3:03
@Email: 2909981736@qq.com
"""
import importlib
import os


def get_class_in_module(class_name, module_path):
    """
    Import a module on the cache directory for modules and extract a class from it.
    """
    module_path = module_path.replace(os.path.sep, ".")
    print(module_path)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
