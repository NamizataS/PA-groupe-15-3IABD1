import os
import numpy as np
from PIL import Image
from ctypes import *
import json
import random
from datetime import datetime

DATASET_FOLDER = "nouveau_dataset"
ACTION_SUBFOLDER = os.path.join(DATASET_FOLDER, "A")
COMEDY_SUBFOLDER = os.path.join(DATASET_FOLDER, "C")
HORROR_SUBFOLDER = os.path.join(DATASET_FOLDER, "H")
path_to_shared_library = "target/release/librust_lib.dylib"


def NameImg(t):
    title = t
    for i in '<>:“/\\|?*.':
        if i in title:
            title = title.replace(i, '')

    title = title.replace(' ', '_')
    return title


def sort_movies(n, length):
    res = []
    if (n <= 0):
        return res
    if (n > length):
        n = length

    while (True):
        s = random.randrange(n)
        if s not in res:
            res.append(s)
        if len(res) == n:
            return res


def get_train_dataset(dataset, n):
    train_dataset = []
    index = sort_movies(n, len(dataset))
    for i in index:
        train_dataset.append(dataset[i])
    return train_dataset, index


def get_test_dataset(index, dataset):
    test_dataset = []
    for i in range(0, len(dataset)):
        if i not in index:
            test_dataset.append(dataset[i])
    return test_dataset


def fill_x_and_y(folder_output, folder_input, dataset):
    with open(folder_output) as f:
        for line in f:
            if line[0] != '[' and line[0] != ']':
                lineJson = json.loads(line[:-2])
                title = NameImg(lineJson["title"])
                file_path = os.path.join(folder_input, f"{title}.png")
                if os.path.isfile(file_path):
                    image = Image.open(file_path)
                    image = image.resize((8, 8))
                    im_arr = np.array(image).flatten()
                    new_row = {'image': np.array(im_arr) / 255.0, 'genre': lineJson['genre']}
                    dataset.append(new_row)


def import_dataset(file_path, folder):
    dataset = []
    fill_x_and_y(file_path, folder, dataset)
    # fill_x_and_y('nouveau_dataset/Comedy.txt', COMEDY_SUBFOLDER, dataset)
    # fill_x_and_y('nouveau_dataset/Horreur.txt', HORROR_SUBFOLDER, dataset)
    return dataset


def toList(arr):
    return [j for i in arr for j in i]


def accuracy(predicted_outputs, y_test):
    sum_rslt = 0
    for i in range(0, len(y_test)):
        if predicted_outputs[i] == y_test[i]:
            sum_rslt += 1
    return sum_rslt / len(y_test)


if __name__ == "__main__":
    my_lib = cdll.LoadLibrary(path_to_shared_library)
    dataset_action = import_dataset('nouveau_dataset/Action.txt', ACTION_SUBFOLDER)
    dataset_comedy = import_dataset('nouveau_dataset/Comedy.txt', COMEDY_SUBFOLDER)
    dataset_horror = import_dataset('nouveau_dataset/Horreur.txt', HORROR_SUBFOLDER)

    train_dataset_action, index_action = get_train_dataset(dataset_action, 293)
    test_dataset_action = get_test_dataset(index_action, dataset_action)

    train_dataset_comedy, index_comedy = get_train_dataset(dataset_comedy, 380)
    test_dataset_comedy = get_test_dataset(index_comedy, dataset_comedy)

    train_dataset_horror, index_horror = get_train_dataset(dataset_horror, 439)
    test_dataset_horror = get_test_dataset(index_horror, dataset_horror)

    train_dataset = train_dataset_action + train_dataset_comedy + train_dataset_horror
    test_dataset = test_dataset_action + test_dataset_comedy + test_dataset_horror

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for image in train_dataset:
        x_train.append(image['image'].tolist())
        y_train.append(image['genre'])
    for image in test_dataset:
        x_test.append(image['image'].tolist())
        y_test.append(image['genre'])

    sample_inputs_len = len(x_train[0])
    x_train = toList(x_train)
    y_train = toList(y_train)

    d = [sample_inputs_len, sample_inputs_len*2, 3]
    arr_size = len(d)
    arr_type = c_int * arr_size
    native_arr = arr_type(*d)

    # create model
    my_lib.create_mlp_model.argtypes = [arr_type, c_int]
    my_lib.create_mlp_model.restype = c_void_p
    model = my_lib.create_mlp_model(native_arr, arr_size)
    print(model)
    # train model
    x_len = len(x_train)
    y_len = len(y_train)

    inputs_type = c_float * x_len
    outputs_type = c_float * y_len

    my_lib.train_classification_stochastic_backdrop_mlp_model.argtypes = [c_void_p, inputs_type, outputs_type, c_float,
                                                                          c_int, c_int, c_int]
    my_lib.train_classification_stochastic_backdrop_mlp_model.restype = None
    my_lib.train_classification_stochastic_backdrop_mlp_model(model, inputs_type(*x_train), outputs_type(*y_train),
                                                              0.0001, 100000, x_len, y_len)

    # predict for the test dataset
    sample_inputs_type = c_float * sample_inputs_len
    my_lib.predict_mlp_model_classification.argtypes = [c_void_p, sample_inputs_type, c_int]
    my_lib.predict_mlp_model_classification.restype = POINTER(c_float)
    predicted_outputs = []
    for p in x_test:
        prediction = my_lib.predict_mlp_model_classification(model, sample_inputs_type(*p), sample_inputs_len)
        prediction = np.ctypeslib.as_array(prediction, (3,))
        predicted_outputs.append(prediction.tolist())

    accuracy_model = accuracy(predicted_outputs, y_test)
    print(f"So the accuracy is {accuracy_model}")
    # print(f"So the predicted outputs are {predicted_outputs}")
    # print(f"And the actual outputs are {y_test} and the len is {len(y_test)}")

    # save model
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    my_lib.save_mlp_model.argtypes = [c_void_p, c_char_p]
    my_lib.save_mlp_model.restype = None
    my_lib.save_mlp_model(model, f"test/MLP/model_mlp_dataset_{dt_string}.json".encode("utf-8"))
    # destroy
    my_lib.destroy_model.argtypes = [c_void_p]
    my_lib.destroy_model.restype = None
    my_lib.destroy_model(model)