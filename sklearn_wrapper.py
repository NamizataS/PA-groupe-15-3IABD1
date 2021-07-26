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


def toList(arr):
    return [j for i in arr for j in i]


class MySKLearnRBFRawWrapper:
    def __init__(self, centers, X, input_dim, gamma, classification: bool = True,alpha: float = 0.01, iteration_count: int = 1000):
        self.lib = get_lib()
        if not hasattr(X, 'shape'):
            X = np.array(X, dtype="object")
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        X = X.flatten()
        X = X.tolist()
        X = toList(X)
        inputs_type = c_float * len(X)
        self.lib.create_rbf_model.argtypes = [c_int, inputs_type, c_int, c_int, c_float]
        self.lib.create_rbf_model.restype = c_void_p
        self.classification = classification
        self.model = self.lib.create_rbf_model(centers, inputs_type(*X), len(X), input_dim, gamma)
        self.alpha = alpha
        self.iteration_count = iteration_count
        self.input_dim = input_dim

    def lloyd(self, X):
        if not hasattr(X, 'shape'):
            X = np.array(X, dtype="object")
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        X = X.flatten()
        X = X.tolist()
        X = toList(X)
        x_type = c_float * len(X)
        self.lib.lloyd.argtypes = [c_void_p, x_type, c_int]
        self.lib.lloyd.restype = None
        self.lib.lloyd(self.model, x_type(*X), len(X))

    def fit(self, X, Y):
        if not hasattr(X, 'shape'):
            X = np.array(X, dtype="object")
        if not hasattr(Y, 'shape'):
            Y = np.array(Y)
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, axis=0)
        X = X.flatten()
        X = X.tolist()
        X = toList(X)
        Y = Y.flatten()
        Y = Y.tolist()
        inputs_type = c_float * len(X)
        outputs_type = c_float * len(Y)
        inputs_native = inputs_type(*X)
        outputs_native = outputs_type(*Y)
        self.lib.train_rbf_model_classification.argtypes = [c_void_p, inputs_type, outputs_type, c_int, c_int, c_int,
                                                            c_float]
        self.lib.train_rbf_model_classification.restype = None
        self.lib.train_rbf_model_regression.argtypes = [c_void_p, inputs_type, outputs_type, c_int, c_int, c_int]
        self.lib.train_rbf_model_regression.restype = None
        if self.classification:
            self.lib.train_rbf_model_classification(self.model, inputs_native, outputs_native, len(X), len(Y),
                                                    self.iteration_count, self.alpha)
        else:
            self.lib.train_rbf_model_regression(self.model, inputs_native, outputs_native, len(X), len(Y), 1)

    def predict(self, X):
        if not hasattr(X, 'shape'):
            X = np.array(X, dtype="object")
        results = []
        sample_inputs_type = c_float * self.input_dim
        self.lib.predict_rbf_model_classification.argtypes = [c_void_p, sample_inputs_type]
        self.lib.predict_rbf_model_classification.restype = c_float
        for x in X:
            results.append(
                self.lib.predict_rbf_model_classification(self.model, sample_inputs_type(*x)))
        return np.array(results)

    def save(self, filename):
        self.lib.save_rbf_model.argtypes = [c_void_p, c_char_p]
        self.lib.save_rbf_model.restype = None
        self.lib.save_rbf_model(self.model, filename.encode("utf-8"))

    def destroy(self):
        self.lib.destroy_rbf_model.argtypes = [c_void_p]
        self.lib.destroy_rbf_model.restype = None
        self.lib.destroy_rbf_model(self.model)

class MySKLearnLinearModelWrapper:
    def __init__(self, array_size: int, iteration_count: int = 1000, alpha=0.01, classification: bool = True):
        self.lib = get_lib()
        self.lib.create_model.argtypes = [c_int]
        self.lib.create_model.restype = POINTER(c_float)
        self.model = self.lib.create_model(array_size)
        self.model_size = array_size + 1
        self.iteration_count = iteration_count
        self.alpha = alpha
        self.classification = classification

    def fit(self, X, Y, inputs_dim: int = 1, outputs_dim: int = 1):
        if not hasattr(X, 'shape'):
            X = np.array(X, dtype="object")
        if not hasattr(Y, 'shape'):
            Y = np.array(Y, dtype="object")
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, axis=0)
        X = X.flatten()
        X = X.tolist()
        X = toList(X)
        Y = Y.flatten()
        Y = Y.tolist()
        inputs_type = c_float * len(X)
        outputs_type = c_float * len(Y)
        inputs_native = inputs_type(*X)
        outputs_native = outputs_type(*Y)
        self.lib.train_rosenblatt_linear_model.argtypes = [type(self.model), inputs_type, outputs_type, c_int, c_float, c_int, c_int, c_int]
        self.lib.train_rosenblatt_linear_model.restype = None
        self.lib.train_regression_linear_model.argtypes = [type(self.model), inputs_type, outputs_type, c_int, c_int, c_int, c_int, c_int]
        self.lib.train_regression_linear_model.restype = None
        if self.classification:
            self.lib.train_rosenblatt_linear_model(self.model, inputs_native, outputs_native, self.iteration_count, self.alpha, self.model_size, len(X), len(Y))
        else:
            self.lib.train_regression_linear_model(self.model, inputs_native, outputs_native, self.model_size, len(X), len(Y), inputs_dim, outputs_dim)

    def predict(self, X, input_dim):
        if not hasattr(X, 'shape'):
            X = np.array(X, dtype="object")
        results = []
        sample_inputs_type = c_float * input_dim
        self.lib.predict_linear_model_classification.argtypes = [type(self.model), sample_inputs_type, c_int, c_int]
        self.lib.predict_linear_model_classification.restype = c_float
        self.lib.predict_linear_model_regression.argtypes = [type(self.model), sample_inputs_type, c_int, c_int]
        self.lib.predict_linear_model_regression.restype = c_float
        if self.classification:
            for x in X:
                prediction = self.lib.predict_linear_model_classification(self.model, sample_inputs_type(*x), self.model_size, input_dim)
                results.append(prediction)
        else:
            for x in X:
                prediction = self.lib.predict_linear_model_regression(self.model, sample_inputs_type(*x), self.model_size, input_dim)
                results.append(prediction)
        return np.array(results)
    def save(self, filename):
        self.lib.save_linear_model.argtypes = [type(self.model), c_int, c_char_p]
        self.lib.save_linear_model.restype = None
        self.lib.save_linear_model(self.model, self.model_size, filename.encode("utf-8"))

    def destroy(self):
        self.lib.destroy_array.argtypes = [type(self.model), c_int]
        self.lib.destroy_array.restype = None
        self.lib.destroy_array(self.model, self.model_size)

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
            X = np.array(X, dtype="object")
        if not hasattr(Y, 'shape'):
            Y = np.array(Y, dtype="object")
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, axis=0)
        X = X.flatten()
        X = X.tolist()
        X = toList(X)
        Y = Y.flatten()
        Y = Y.tolist()
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
            X = np.array(X, dtype="object")
        results = []
        sample_inputs_type = c_float * input_dim
        self.lib.predict_mlp_model_classification.argtypes = [c_void_p, sample_inputs_type, c_int]
        self.lib.predict_mlp_model_classification.restype = POINTER(c_float)
        for x in X:
            prediction = self.lib.predict_mlp_model_classification(self.model, sample_inputs_type(*x), input_dim)
            result = np.ctypeslib.as_array(prediction, (output_dim,))
            results.append(np.array(result))
        return np.array(results)

    def save(self, filename):
        self.lib.save_mlp_model.argtypes = [c_void_p, c_char_p]
        self.lib.save_mlp_model.restype = None
        self.lib.save_mlp_model(self.model, filename.encode("utf-8"))

    def destroy(self):
        self.lib.destroy_model.argtypes = [c_void_p]
        self.lib.destroy_model.restype = None
        self.lib.destroy_model(self.model)
