import os
import numpy as np
from PIL import Image
from ctypes import *
import json
import random
from sklearn_wrapper import MySKLearnMLPRawWrapper
from datetime import datetime
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

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
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    #y_train = [[i[0], i[2]] for i in y_train]
    #y_test = [[i[0], i[2]] for i in y_test]
    sample_inputs_len = len(x_train[0])
    d = [sample_inputs_len, sample_inputs_len,2]
    losses = []
    val_losses = []
    accs = []
    val_accs = []
    wrapped_model = MySKLearnMLPRawWrapper(d, alpha=0.1,iteration_count=len(x_train))
    for epoch in tqdm.tqdm(range(1000)):
        wrapped_model.fit(x_train, y_train)
        predicted_y_train = wrapped_model.predict(x_train, sample_inputs_len, 3)
        predicted_y_test = wrapped_model.predict(x_test, sample_inputs_len, 3)

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
    plt.title('Evolution of loss (MSE) for MLP 192, 192, 3')
    plt.xlabel('epochs')
    plt.ylabel(f'mean squared error')
    plt.savefig(f'test_dataset/fig/fig_MLP_loss_{dt_string}.png')
    plt.show()

    plt.plot(accs)
    plt.plot(val_accs)
    plt.legend(['acc', 'val_acc'], loc='upper left')
    plt.title('Evolution of accuracy for MLP 192, 192, 3')
    plt.xlabel('epochs')
    plt.ylabel(f'accuracy')
    plt.savefig(f'test_dataset/fig/fig_MLP_accuracy_{dt_string}.png')
    plt.show()

    #save
    wrapped_model.save(f"test_dataset/MLP/model_mlp_dataset_two_class_{dt_string}.json")

    #destroy
    wrapped_model.destroy()
