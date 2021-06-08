from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import os

path_to_shared_library = "target/debug/librust_lib.dylib"
def toList(arr):
    return [j for i in arr for j in i]

if __name__ == "__main__":
    my_lib = cdll.LoadLibrary(path_to_shared_library)
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    Y = np.array([
        2,
        1,
        -2,
        -1
    ])

    d = [2, 2, 1]
    d_len = len(d)
    d_type = c_int * d_len
    my_lib.create_mlp_model.argtypes = [d_type, c_int]
    my_lib.create_mlp_model.restype = c_void_p

    model = my_lib.create_mlp_model(d_type(*d), d_len)
    sample_inputs_len = 2
    sample_inputs_type = c_float * sample_inputs_len

    my_lib.predict_mlp_model_regression.argtypes = [c_void_p, sample_inputs_type, c_int]
    my_lib.predict_mlp_model_regression.restype = POINTER(c_float)
    test_dataset_inputs = [[i,j] for i in range(-10, 11) for j in range (-10,11)]
    predicted_outputs = [my_lib.predict_mlp_model_regression(model, sample_inputs_type(*p), sample_inputs_len)[0] for p
                         in test_dataset_inputs]
    test_dataset_inputs = np.array(test_dataset_inputs)

    # Free memory
    my_lib.destroy_model.argtypes = [c_void_p]
    my_lib.destroy_model.restype = None
    my_lib.destroy_model(model)