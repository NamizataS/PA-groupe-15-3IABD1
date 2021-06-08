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
        [1],
        [2]
    ])
    Y = np.array([
        2,
        3
    ])

    dataset_inputs = toList(X)
    dataset_expected_outputs = Y.tolist()

    my_lib.create_model.argtypes = [c_int]
    my_lib.create_model.restype = POINTER(c_float)

    arr_size = 1
    native_arr = my_lib.create_model(arr_size)

    arr_size += 1

    inputs_len = len(dataset_inputs)
    expected_len = len(dataset_expected_outputs)
    arr_type = type(native_arr)
    inputs_type = c_float * inputs_len
    outputs_type = c_float * expected_len

    # Declaration of argtypes and restypes for the function predict_linear_model_classification
    coordinate_type = c_float * 1
    my_lib.predict_linear_model_regression.argtypes = [arr_type, coordinate_type, c_int, c_int]
    my_lib.predict_linear_model_regression.restype = c_float

    plt.axis([-10, 10, -10, 10])
    test_dataset = [[i] for i in range(-10, 11)]
    predicted_outputs = [
        my_lib.predict_linear_model_regression(native_arr, coordinate_type(*point), arr_size, len(point)) for point in
        test_dataset]

    plt.scatter(test_dataset, predicted_outputs)
    plt.scatter(dataset_inputs, dataset_expected_outputs, c="purple")
    plt.show()

    my_lib.train_regression_linear_model.argtypes = [arr_type, inputs_type, outputs_type, c_int, c_int, c_int]
    my_lib.train_regression_linear_model.restype = None
    inputs_native = inputs_type(*dataset_inputs)
    outputs_native = outputs_type(*dataset_expected_outputs)

    my_lib.train_regression_linear_model(native_arr, inputs_native, outputs_native, arr_size, inputs_len, expected_len)
    np_arr = np.ctypeslib.as_array(native_arr, (arr_size,))

    plt.axis([-10, 10, -10, 10])
    predicted_outputs = [
        my_lib.predict_linear_model_regression(native_arr, coordinate_type(*point), arr_size, len(point)) for point in
        test_dataset]

    plt.scatter(test_dataset, predicted_outputs)
    plt.scatter(dataset_inputs, dataset_expected_outputs, c="purple")
    plt.show()

    my_lib.destroy_array.argtypes = [POINTER(c_float), c_int]
    my_lib.destroy_array.restype = None
    my_lib.destroy_array(native_arr, arr_size)
