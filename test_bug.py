import os
import random
import numpy as np
from PIL import Image
import json
import dashboard
from ctypes import *
import matplotlib.pyplot as plt

path_to_shared_library = "target/release/librust_lib.dylib"
def toList(arr):
    return [j for i in arr for j in i]

if __name__ == '__main__':
    my_lib = cdll.LoadLibrary(path_to_shared_library)
    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    # Convert array to list
    dataset_inputs = toList(X)
    dataset_expected_outputs = toList(Y)
    inputs_len = len(dataset_inputs)
    expected_len = len(dataset_expected_outputs)
    inputs_dim = inputs_len // expected_len

    # Declaration of argtypes and restypes for the function create_model_SVM
    inputs_type = c_float * inputs_len
    outputs_type = c_float * expected_len
    my_lib.train_model_SVM.argtypes = [inputs_type, outputs_type, c_int, c_int, c_int]
    my_lib.train_model_SVM.restype = POINTER(c_double)

    inputs_native = inputs_type(*dataset_inputs)
    outputs_native = outputs_type(*dataset_expected_outputs)

    # Creation of model
    native_arr = my_lib.train_model_SVM(inputs_native, outputs_native, inputs_len, expected_len, inputs_dim)
    model_len = inputs_dim + 1

    # Declaration of argtypes and restypes for the function predict_SVM
    native_type = type(native_arr)
    coordonate_type = c_double * model_len
    my_lib.predict_SVM.argtypes = [native_type, coordonate_type, c_int]
    my_lib.predict_SVM.restype = c_float

    # Trainning
    r = np.arange(0.0, 4.0, 0.25)

    points_x1_blue = []
    points_x2_blue = []

    points_x1_red = []
    points_x2_red = []

    for i in r:
        for j in r:
            coord = coordonate_type(*[1.0, i, j])
            res = my_lib.predict_SVM(native_arr, coord, model_len)

            if res > 0.0:
                points_x1_blue.append(i)
                points_x2_blue.append(j)
            else:
                points_x1_red.append(i)
                points_x2_red.append(j)

    plt.scatter(points_x1_blue, points_x2_blue, c='blue')
    plt.scatter(points_x1_red, points_x2_red, c='red')

    plt.scatter([X[p][0] for p in range(len(Y)) if Y[p] > 0], [X[p][1] for p in range(len(Y)) if Y[p] > 0], c='blue',
                s=100)
    plt.scatter([X[p][0] for p in range(len(Y)) if Y[p] < 0], [X[p][1] for p in range(len(Y)) if Y[p] < 0], c='red',
                s=100)

    plt.show()
    plt.clf()

    # Free memory
    my_lib.destroy_array_double.argtypes = [POINTER(c_double), c_int]
    my_lib.destroy_array_double.restype = None
    my_lib.destroy_array_double(native_arr, model_len)