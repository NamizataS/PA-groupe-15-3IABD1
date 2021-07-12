from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

path_to_shared_library = "target/release/librust_lib.dylib"
def toList(arr):
    return [j for i in arr for j in i]

if __name__ == "__main__":
    my_lib = cdll.LoadLibrary(path_to_shared_library)
    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    # Convert array to list
    dataset_inputs = toList(X)
    dataset_expected_outputs = Y
    inputs_len = len(dataset_inputs)
    outputs_len = len(dataset_expected_outputs)
    output_dim = 1

    dataset_inputs_type = c_float * inputs_len
    dataset_inputs_native = dataset_inputs_type(*dataset_inputs)
    dataset_outputs_type = c_float * outputs_len
    dataset_outputs_native = dataset_outputs_type(*dataset_expected_outputs)
    sample_inputs_type = c_float * 2
    iterations = 40000

    # create model
    my_lib.create_rbf_model.argtypes = [c_int, dataset_inputs_type, c_int, c_int]
    my_lib.create_rbf_model.restype = c_void_p

    model = my_lib.create_rbf_model(10, dataset_inputs_native, inputs_len, len(X[0]))

    # test dataset
    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 40, 2) for x2 in range(-10, 40, 2)]
    colors = ["blue" if output >= 0 else "red" for output in dataset_expected_outputs]

    # training
    my_lib.lloyd.argtypes = [c_void_p, dataset_inputs_type, c_int, c_int]
    my_lib.lloyd.restype = None
    my_lib.lloyd(model, dataset_inputs_native, inputs_len, iterations)

    my_lib.train_rbf_model_classification.argtypes = [c_void_p, dataset_inputs_type, dataset_outputs_type, c_int, c_int,
                                                      c_int, c_float]
    my_lib.train_rbf_model_classification.restype = POINTER(c_float)
    loss = my_lib.train_rbf_model_classification(model, dataset_inputs_native, dataset_outputs_native, inputs_len, outputs_len,
                                          iterations, 0.0001)
    loss_arr = np.ctypeslib.as_array(loss, (iterations,))

    #graph for the loss
    indexes = np.arange(iterations)
    plt.plot(indexes, loss_arr, c='red')
    # permet d'afficher la zone directement de façon plus lisible
    # affiche la moyenne
    plt.legend(['mean', 'std'], loc='upper left')  # légende du graphe
    plt.title('loss on linear multiple problem')  # titre du graphe
    plt.xlabel('epochs')  # label de l'axe x
    plt.show()
    # destroy model
    my_lib.destroy_rbf_model.argtypes = [c_void_p]
    my_lib.destroy_rbf_model.restype = None
    my_lib.destroy_rbf_model(model)