from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import os

path_to_shared_library = "target/debug/librust_lib.dylib"
def toList(arr):
    return [j for i in arr for j in i]

if __name__ == "__main__":
    my_lib = cdll.LoadLibrary(path_to_shared_library)
    # Dataset
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

    d = [2, 1]

    # Convert array to list
    dataset_inputs = toList(X)
    dataset_expected_outputs = Y.tolist()
    inputs_len = len(dataset_inputs)
    expected_len = len(dataset_expected_outputs)

    # Declaration of argtypes and restypes for the function create_mlp_model
    arr_size = len(d)
    arr_type = c_int * arr_size
    native_arr = arr_type(*d)
    my_lib.create_mlp_model.argtypes = [arr_type, c_int]
    my_lib.create_mlp_model.restype = c_void_p

    model = my_lib.create_mlp_model(native_arr, 2)
    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 40, 2) for x2 in range(-10, 40, 2)]
    colors = ["blue" if output >= 0 else "red" for output in dataset_expected_outputs]

    # Declaration of argtypes and restypes for the function predict_mlp_model_classification
    p_type = c_float * 2
    my_lib.predict_mlp_model_classification.argtypes = [c_void_p, p_type, c_int]
    my_lib.predict_mlp_model_classification.restype = POINTER(c_float)
    p_len = 2

    predicted_outputs = [my_lib.predict_mlp_model_classification(model, p_type(*p), p_len)[0] for p in test_dataset]
    predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()

    flattened_len = len(dataset_inputs)
    # Declaration of argtypes and restypes for the function predict_mlp_model_classification
    inputs_type = c_float * flattened_len
    outputs_type = c_float * expected_len
    my_lib.train_classification_stochastic_backdrop_mlp_model.argtypes = [c_void_p, inputs_type, outputs_type, c_float,
                                                                          c_int, c_int, c_int]
    my_lib.train_classification_stochastic_backdrop_mlp_model.restype = None
    inputs_native = inputs_type(*dataset_inputs)
    outputs_native = outputs_type(*dataset_expected_outputs)

    my_lib.train_classification_stochastic_backdrop_mlp_model(model, inputs_native, outputs_native, 0.001, 100000,
                                                              flattened_len, expected_len)

    predicted_outputs = [my_lib.predict_mlp_model_classification(model, p_type(*p), p_len)[0] for p in test_dataset]
    predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()

    # Free memory
    my_lib.destroy_model.argtypes = [c_void_p]
    my_lib.destroy_model.restype = None
    my_lib.destroy_model(model)