import numpy as np
import plotly.graph_objects as go
from ctypes import *
import platform
import matplotlib.pyplot as plt


def declare_lib():
    path_to_shared_library = "target/release/librust_lib.dylib"
    path_to_shared_library_windows = "target\\debug\\rust_lib.dll"

    if (platform.system() == 'Windows'):
        my_lib = cdll.LoadLibrary(path_to_shared_library_windows)
    else:
        my_lib = cdll.LoadLibrary(path_to_shared_library)
    return my_lib

def accuracy(predicted_outputs, y_test):
    sum_rslt = 0
    for i in range(0, len(y_test)):
        if predicted_outputs[i] == y_test[i]:
            sum_rslt += 1
    return sum_rslt / len(y_test)

def dash_rbf():
    my_lib = declare_lib()
    my_lib.load_rbf_model.argtypes = [c_char_p]
    my_lib.load_rbf_model.restype = c_void_p
    iterations = 10000
    model_action = my_lib.load_rbf_model('test_dataset/RBF/model_action_rbf_multiclass_17_07_2021_17_01.json'.encode("utf-8"))
    model_comedy = my_lib.load_rbf_model('test_dataset/RBF/model_comedy_rbf_multiclass_17_07_2021_17_01.json'.encode("utf-8"))
    model_horror = my_lib.load_rbf_model('test_dataset/RBF/model_horror_rbf_multiclass_17_07_2021_17_01.json'.encode("utf-8"))
