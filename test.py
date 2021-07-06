from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

path_to_shared_library = "target/debug/librust_lib.dylib"
def toList(arr):
    return [j for i in arr for j in i]

if __name__ == "__main__":
    my_lib = cdll.LoadLibrary(path_to_shared_library)
    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])
    Y = np.array([
        1,
        -1,
        -1
    ])

    # Convert array to list
    dataset_inputs = toList(X)
    dataset_expected_outputs = Y.tolist()
    inputs_len = len(dataset_inputs)
    outputs_len = len(dataset_expected_outputs)
    output_dim = 1

    dataset_inputs_type = c_float * inputs_len
    dataset_inputs_native = dataset_inputs_type(*dataset_inputs)
    dataset_outputs_type = c_float * outputs_len
    dataset_outputs_native = dataset_outputs_type(*dataset_expected_outputs)
    sample_inputs_type = c_float * len(X[0])

    # create model
    my_lib.create_rbf_model.argtypes = [c_int, dataset_inputs_type, c_int, c_int]
    my_lib.create_rbf_model.restype = c_void_p

    model = my_lib.create_rbf_model(3, dataset_inputs_native, inputs_len, len(X[0]))

    # test dataset
    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 40, 2) for x2 in range(-10, 40, 2)]
    colors = ["blue" if output >= 0 else "red" for output in dataset_expected_outputs]  # destroy model

    # before training
    my_lib.predict_rbf_model_classification.argtypes = [c_void_p, sample_inputs_type]
    my_lib.predict_rbf_model_classification.restype = c_float
    predicted_outputs = [my_lib.predict_rbf_model_classification(model, sample_inputs_type(*p)) for p in test_dataset]
    predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()

    # training
    my_lib.lloyd2.argtypes = [c_void_p, dataset_inputs_type, c_int]
    my_lib.lloyd2.restype = None
    my_lib.lloyd2(model, dataset_inputs_native, inputs_len)

    my_lib.train_rbf_model_classification.argtypes = [c_void_p, dataset_inputs_type, dataset_outputs_type, c_int, c_int,
                                                      c_int, c_float]
    my_lib.train_rbf_model_classification.restype = None
    my_lib.train_rbf_model_classification(model, dataset_inputs_native, dataset_outputs_native, inputs_len, outputs_len,
                                          100000, 0.0001)

    # after training
    predicted_outputs = [my_lib.predict_rbf_model_classification(model, sample_inputs_type(*p)) for p in test_dataset]
    predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()

    # destroy model
    my_lib.destroy_rbf_model.argtypes = [c_void_p]
    my_lib.destroy_rbf_model.restype = None
    my_lib.destroy_rbf_model(model)