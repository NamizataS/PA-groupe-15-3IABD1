from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import os

path_to_shared_library = "target/debug/librust_lib.dylib"

if __name__ == "__main__":
    my_lib = cdll.LoadLibrary(path_to_shared_library)
    d = [2, 2, 1]
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])

    X_test = [2.0, -3.0]

    X_flat = []
    for elt in X:
        X_flat.append(elt[0])
        X_flat.append(elt[1])

    arr_size = len(d)
    arr_type = c_int * arr_size
    native_arr = arr_type(*d)
    my_lib.create_mlp_model.argtypes = [arr_type, c_int]
    my_lib.create_mlp_model.restype = c_void_p

    mlp = my_lib.create_mlp_model(native_arr, arr_size)
    x_size = len(X_flat)
    x_type = c_float * x_size
    y_size = len(Y)
    y_type = c_float * y_size
    my_lib.train_classification_stochastic_backdrop_mlp_model.argtypes = [c_void_p, x_type, y_type, c_float, c_int,
                                                                          c_int, c_int]
    my_lib.train_classification_stochastic_backdrop_mlp_model.restype = None
    x_native = x_type(*X_flat)
    y_native = y_type(*Y)
    my_lib.train_classification_stochastic_backdrop_mlp_model(mlp, x_native, y_native, 0.01, 100000, x_size, y_size)

    x_test_size = len(X_test)
    x_test_type = c_float * x_test_size
    my_lib.predict_mlp_model_classification.argtypes = [c_void_p, x_test_type, c_int]
    my_lib.predict_mlp_model_classification.restype = POINTER(c_float)
    x_test_native = x_test_type(*X_test)
    pred = my_lib.predict_mlp_model_classification(mlp, x_test_native, x_test_size)
    np_arr = np.ctypeslib.as_array(pred, (1,))
    print(np_arr)

    my_lib.destroy_model.argtypes = [c_void_p]
    my_lib.destroy_model.restype = None
    my_lib.destroy_model(mlp)
