import os
import numpy as np
from PIL import Image
import platform
from ctypes import *
import matplotlib.pyplot as plt

DATASET_FOLDER = "dataset"
TRAIN_SUBFOLDER = os.path.join(DATASET_FOLDER, "train")
TEST_SUBFOLDER = os.path.join(DATASET_FOLDER, "test")

TRAIN_ACTION_FOLDER = os.path.join(TRAIN_SUBFOLDER, "action")
TRAIN_COMEDY_FOLDER = os.path.join(TRAIN_SUBFOLDER, "comedie")

TEST_ACTION_FOLDER = os.path.join(TEST_SUBFOLDER, "action")
TEST_COMEDY_FOLDER = os.path.join(TEST_SUBFOLDER, "comedie")


def fill_x_and_y(folder, x_list, y_list, label):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        image = Image.open(file_path)
        image = image.resize((32, 32))
        im_arr = np.array(image).flatten()
        x_list.append(im_arr)
        y_list.append(label)


def import_dataset():
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    fill_x_and_y(TRAIN_ACTION_FOLDER, X_train, Y_train, 1)
    fill_x_and_y(TRAIN_COMEDY_FOLDER, X_train, Y_train, 0)
    fill_x_and_y(TEST_ACTION_FOLDER, X_test, Y_test, 1)
    fill_x_and_y(TEST_COMEDY_FOLDER, X_test, Y_test, 0)

    return (np.array(X_train).astype(np.float), np.array(Y_train).astype(np.float)), (
        np.array(X_test).astype(np.float), np.array(Y_test).astype(np.float))


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = import_dataset()
    path_to_shared_library = "target/debug/librust_lib.dylib"
    path_to_shared_library_windows = "target\\debug\\rust_lib.dll"

    if platform.system() == 'Windows':
        my_lib = cdll.LoadLibrary(path_to_shared_library_windows)
    else:
        my_lib = cdll.LoadLibrary(path_to_shared_library)

    d = [2, 4, 4, 1]
    arr_size = len(d)
    arr_type = c_int * arr_size
    native_arr = arr_type(*d)
    my_lib.create_mlp_model.argtypes = [arr_type, c_int]
    my_lib.create_mlp_model.restype = c_void_p

    model = my_lib.create_mlp_model(native_arr, arr_size)

    X_train = X_train.flatten()
    Y_train = Y_train.flatten()

    flattened_inputs_len = len(X_train)
    print(flattened_inputs_len)
    flattened_outputs_len = len(Y_train)

    inputs_type = c_float * flattened_inputs_len
    outputs_type = c_float * flattened_outputs_len

    my_lib.train_classification_stochastic_backdrop_mlp_model.argtypes = [c_void_p, inputs_type, outputs_type, c_float,
                                                                          c_int, c_int, c_int]
    my_lib.train_classification_stochastic_backdrop_mlp_model.restype = None
    inputs_native = inputs_type(*X_train)
    outputs_native = outputs_type(*Y_train)

    my_lib.train_classification_stochastic_backdrop_mlp_model(model, inputs_native, outputs_native, 0.001, 100000,
                                                              flattened_inputs_len, flattened_outputs_len)