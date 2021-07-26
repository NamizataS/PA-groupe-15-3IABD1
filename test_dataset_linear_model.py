import os
import random
import numpy as np
from PIL import Image
import json
from ctypes import *
from sklearn_wrapper import MySKLearnLinearModelWrapper
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from datetime import datetime

DATASET_FOLDER = "nouveau_dataset"
ACTION_SUBFOLDER = os.path.join(DATASET_FOLDER, "A")
COMEDY_SUBFOLDER = os.path.join(DATASET_FOLDER, "C")
HORROR_SUBFOLDER = os.path.join(DATASET_FOLDER, "H")

path_to_shared_library = "target/release/librust_lib.dylib"


def import_dataset(file_path):
    with open(file_path) as json_file:
        dataset = json.load(json_file)
    return dataset


def toList(arr):
    return [j for i in arr for j in i]


def get_pixels(image):
    image = Image.open(image)
    image = image.resize((8, 8))
    im_arr = np.array(image).flatten()
    return np.array(im_arr) / 255.0

if __name__ == "__main__":
    my_lib = cdll.LoadLibrary(path_to_shared_library)

    train_dataset = import_dataset('nouveau_dataset/train_dataset.json')
    test_dataset = import_dataset('nouveau_dataset/test_dataset.json')
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for image in train_dataset:
        x_train.append(get_pixels(image['image']))
        y_train.append(image['genre'])
    for image in test_dataset:
        x_test.append(get_pixels(image['image']))
        y_test.append(image['genre'])
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)
    sample_inputs_len = len(x_train[0])
    y_train_action = y_train[:, 1]
    y_train_comedy = y_train[:, 2]
    y_train_horror = y_train[:, 0]
    #y_train = [[i[0], i[2]] for i in y_train]
    #y_test = [[i[0], i[2]] for i in y_test]
    inputs_len = len(x_train)
    outputs_len = len(y_train_horror)
    output_dim = 1

    wrapped_model_action = MySKLearnLinearModelWrapper(sample_inputs_len)
    wrapped_model_comedy = MySKLearnLinearModelWrapper(sample_inputs_len)
    wrapped_model_horror = MySKLearnLinearModelWrapper(sample_inputs_len)
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    for epoch in tqdm.tqdm(range(1000)):
        wrapped_model_action.fit(x_train, y_train_action)
        wrapped_model_comedy.fit(x_train, y_train_comedy)
        wrapped_model_horror.fit(x_train, y_train_horror)

        predicted_y_train_action = wrapped_model_action.predict(x_train, sample_inputs_len)
        predicted_y_train_comedy = wrapped_model_comedy.predict(x_train, sample_inputs_len)
        predicted_y_train_horror = wrapped_model_horror.predict(x_train, sample_inputs_len)

        predicted_y_train = []
        for i in range(0, len(predicted_y_train_horror)):
            predicted_y_train.append(
                [predicted_y_train_horror[i], predicted_y_train_action[i], predicted_y_train_comedy[i]])

        predicted_y_test_action = wrapped_model_action.predict(x_test, sample_inputs_len)
        predicted_y_test_comedy = wrapped_model_comedy.predict(x_test, sample_inputs_len)
        predicted_y_test_horror = wrapped_model_horror.predict(x_test, sample_inputs_len)

        predicted_y_test = []
        for i in range(0, len(predicted_y_test_horror)):
            predicted_y_test.append(
                [predicted_y_test_horror[i], predicted_y_test_action[i], predicted_y_test_comedy[i]])

        loss = mean_squared_error(y_train, predicted_y_train)
        losses.append(loss)
        val_loss = mean_squared_error(y_test, predicted_y_test)
        val_losses.append(val_loss)

        acc = accuracy_score(np.argmax(y_train, axis=1), np.argmax(predicted_y_train, axis=1))
        accs.append(acc)
        val_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predicted_y_test, axis=1))
        val_accs.append(val_acc)

    print(val_accs[-1])
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.title('Evolution of loss(MSE) linear model image resized (8,8)')
    plt.xlabel('epochs')
    plt.ylabel(f'mean squared error')
    plt.savefig(f'test_dataset/fig/fig_linear_model_loss_{dt_string}.png')
    plt.show()

    plt.plot(accs)
    plt.plot(val_accs)
    plt.legend(['acc', 'val_acc'], loc='upper left')
    plt.title('Evolution of accuracy linear model image resized (8,8)')
    plt.xlabel('epochs')
    plt.ylabel(f'accuracy')
    plt.savefig(f'test_dataset/fig/fig_linear_model_accuracy_{dt_string}.png')
    plt.show()

    # save model
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    wrapped_model_action.save(f"test_dataset/linear_model/model_action_linear_{dt_string}.json")
    wrapped_model_comedy.save(f"test_dataset/linear_model/model_comedy_linear_{dt_string}.json")
    wrapped_model_horror.save(f"test_dataset/linear_model/model_horror_linear_{dt_string}.json")

    #destroy
    wrapped_model_action.destroy()
    wrapped_model_comedy.destroy()
    wrapped_model_horror.destroy()