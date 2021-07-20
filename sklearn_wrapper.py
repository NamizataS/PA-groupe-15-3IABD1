import numpy as np
import tqdm
import platform
from ctypes import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix


def get_lib():
    path_to_shared_library = "target/release/librust_lib.dylib"
    path_to_shared_library_windows = "target\\debug\\rust_lib.dll"

    if platform.system() == 'Windows':
        my_lib = cdll.LoadLibrary(path_to_shared_library_windows)
    else:
        my_lib = cdll.LoadLibrary(path_to_shared_library)
    return my_lib


class MySKLearnRBFRawWrapper:
    def __init__(self, centers, X, input_dim, gamma, alpha: float = 0.01, iteration_count: int = 1000):
        self.lib = get_lib()
        if not hasattr(X, 'shape'):
            X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        X = X.flatten()
        inputs_type = c_float * len(X)
        self.lib.create_rbf_model.argtypes = [c_int, inputs_type, c_int, c_int, c_float]
        self.lib.create_rbf_model.restype = c_void_p
        self.model = self.lib.create_rbf_model(centers, inputs_type(*X), len(X), input_dim, gamma)
        self.alpha = alpha
        self.iteration_count = iteration_count
        self.input_dim = input_dim

    def lloyd(self, X):
        if not hasattr(X, 'shape'):
            X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        X = X.flatten()
        x_type = c_float * len(X)
        self.lib.lloyd.argtypes = [c_void_p, x_type, c_int]
        self.lib.lloyd.restype = None
        self.lib.lloyd(self.model, x_type(*X), len(X))

    def fit(self, X, Y):
        if not hasattr(X, 'shape'):
            X = np.array(X)
        if not hasattr(Y, 'shape'):
            Y = np.array(Y)
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, axis=0)
        X = X.flatten()
        Y = Y.flatten()
        inputs_type = c_float * len(X)
        outputs_type = c_float * len(Y)
        inputs_native = inputs_type(*X)
        outputs_native = outputs_type(*Y)
        self.lib.train_rbf_model_classification.argtypes = [c_void_p, inputs_type, outputs_type, c_int, c_int, c_int, c_float]
        self.lib.train_rbf_model_classification.restype = None
        self.lib.train_rbf_model_classification(self.model, inputs_native, outputs_native, len(X), len(Y),
                                          self.iteration_count, self.alpha)

    def predict(self, X):
        if not hasattr(X, 'shape'):
            X = np.array(X)
        results = []
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        sample_inputs_type = c_float * self.input_dim
        self.lib.predict_rbf_model_classification.argtypes = [c_void_p, sample_inputs_type]
        self.lib.predict_rbf_model_classification.restype = c_float
        for x in X:
            results.append(
               self.lib.predict_rbf_model_classification(self.model, sample_inputs_type(*x.flatten())))
        return np.array(results)

    def destroy(self):
        self.lib.destroy_rbf_model.argtypes = [c_void_p]
        self.lib.destroy_rbf_model.restype = None
        self.lib.destroy_rbf_model(self.model)


class MySKLearnMLPRawWrapper:
    def __init__(self, npl: [int], classification: bool = True, alpha: float = 0.01, iteration_count: int = 1000):
        self.lib = get_lib()
        arr_type = c_int * len(npl)
        self.lib.create_mlp_model.argtypes = [arr_type, c_int]
        self.lib.create_mlp_model.restype = c_void_p
        self.model = self.lib.create_mlp_model(arr_type(*npl), len(npl))
        self.classification = classification
        self.alpha = alpha
        self.iteration_count = iteration_count

    def fit(self, X, Y):
        if not hasattr(X, 'shape'):
            X = np.array(X)
        if not hasattr(Y, 'shape'):
            Y = np.array(Y)
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, axis=0)
        X = X.flatten()
        Y = Y.flatten()
        inputs_type = c_float * len(X)
        outputs_type = c_float * len(Y)
        inputs_native = inputs_type(*X)
        outputs_native = outputs_type(*Y)
        self.lib.train_classification_stochastic_backdrop_mlp_model.argtypes = [c_void_p, inputs_type, outputs_type,
                                                                                c_float, c_int, c_int, c_int]
        self.lib.train_classification_stochastic_backdrop_mlp_model.restype = None
        self.lib.train_classification_stochastic_backdrop_mlp_model(self.model, inputs_native,
                                                                    outputs_native, self.alpha,
                                                                    self.iteration_count, len(X),
                                                                    len(Y))

    def predict(self, X, input_dim, output_dim):
        if not hasattr(X, 'shape'):
            X = np.array(X)
        results = []
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        sample_inputs_type = c_float * input_dim
        self.lib.predict_mlp_model_classification.argtypes = [c_void_p, sample_inputs_type, c_int]
        self.lib.predict_mlp_model_classification.restype = POINTER(c_float)
        for x in X:
            results.append(
                np.ctypeslib.as_array(
                    self.lib.predict_mlp_model_classification(self.model, sample_inputs_type(*x.flatten()), input_dim),
                    (output_dim,)))
        return np.array(results)

    def destroy(self):
        self.lib.destroy_model.argtypes = [c_void_p]
        self.lib.destroy_model.restype = None
        self.lib.destroy_model(self.model)
