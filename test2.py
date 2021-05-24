from ctypes import *
import numpy as np
import os

path_to_shared_library = "target\\debug\\rust_lib.dll"

if __name__ == "__main__":
    my_lib = cdll.LoadLibrary(os.path.abspath(path_to_shared_library))
    my_lib.create_model.argtypes = [c_int]
    my_lib.create_model.restype = POINTER(c_float)

    arr_size = 2
    native_arr = my_lib.create_model(arr_size)

    dataset_inputs = [1.0, 2.0, 3.0]
    dataset_expected_outputs = [3.0, 5.0, 7.1]
    inputs_len = len(dataset_inputs)
    expected_len = len(dataset_expected_outputs)
    arr_type = type(native_arr)
    inputs_type = c_float * inputs_len
    outputs_type = c_float * expected_len

    my_lib.train_regression_linear_model.argtypes = [arr_type, inputs_type, outputs_type, c_int, c_int, c_int]
    my_lib.train_regression_linear_model.restype =None
    inputs_native = inputs_type(*dataset_inputs)
    outputs_native = outputs_type(*dataset_expected_outputs)

    input_dim = arr_size-1
    output_dim = arr_size-1
    sample_count = inputs_len//input_dim

    my_lib.train_regression_linear_model(native_arr, inputs_native, outputs_native, sample_count,input_dim,output_dim)
    np_arr = np.ctypeslib.as_array(native_arr, (arr_size,))
    print(np_arr)

    my_lib.destroy_array.argtypes = [POINTER(c_float), c_int]
    my_lib.destroy_array.restype = None

    my_lib.destroy_array(native_arr, arr_size)
