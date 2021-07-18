from ctypes import *
import platform
import numpy as np
from PIL import Image
import os


class Predict:
    def __init__(self, image, size):
        self.image_path = image
        arr_image = Image.open(image)
        image = arr_image.resize((size))
        im_arr = np.array(image).flatten()
        im = np.array(im_arr) / 255.0
        self.categories = ['Action', 'Comedy', 'Horror']
        self.image = im.tolist()

    def declare_lib(self):
        path_to_shared_library = "target/release/librust_lib.dylib"
        path_to_shared_library_windows = "target\\debug\\rust_lib.dll"

        if (platform.system() == 'Windows'):
            my_lib = cdll.LoadLibrary(path_to_shared_library_windows)
        else:
            my_lib = cdll.LoadLibrary(path_to_shared_library)
        return my_lib

    def predict_rbf(self):
        my_lib = self.declare_lib()
        my_lib.load_rbf_model.argtypes = [c_char_p]
        my_lib.load_rbf_model.restype = c_void_p
        model_action = my_lib.load_rbf_model('test_keep/model_action_rbf_multiclass_17_07_2021_17_39.json'.encode("utf-8"))
        model_comedy = my_lib.load_rbf_model('test_keep/model_comedy_rbf_multiclass_17_07_2021_17_39.json'.encode("utf-8"))
        model_horror = my_lib.load_rbf_model('test_keep/model_horror_rbf_multiclass_17_07_2021_17_39.json'.encode("utf-8"))
        sample_inputs_len = len(self.image)
        sample_inputs_type = c_float * sample_inputs_len
        sample_inputs_native = sample_inputs_type(*self.image)
        my_lib.predict_rbf_model_classification.argtypes = [c_void_p, sample_inputs_type]
        my_lib.predict_rbf_model_classification.restype = c_float
        prediction = [my_lib.predict_rbf_model_classification(model_action, sample_inputs_native),
                      my_lib.predict_rbf_model_classification(model_comedy, sample_inputs_native),
                      my_lib.predict_rbf_model_classification(model_horror, sample_inputs_native)]
        os.remove(self.image_path)

        predict = [self.categories[i] for i in range(0,len(self.categories)) if prediction[i] == 1]
        return predict

    def predict_mlp(self):
        my_lib = self.declare_lib()
        my_lib.load_mlp_model.argtypes = [c_char_p]
        my_lib.load_mlp_model.restype = c_void_p
        model = my_lib.load_mlp_model('test_keep/model_mlp_dataset.json')
        sample_inputs_len = len(self.image)
        sample_inputs_type = c_float * sample_inputs_len
        my_lib.predict_mlp_model_classification.argtypes = [c_void_p, sample_inputs_type, c_int]
        my_lib.predict_mlp_model_classification.restype = POINTER(c_float)
        prediction = my_lib.predict_mlp_model_classification(model, sample_inputs_type(*self.image), sample_inputs_len)
        prediction = np.ctypeslib.as_array(prediction, (3,))
        predict = [self.categories[i] for i in range(0,len(self.categories)) if prediction[i] == 1]
        return predict
