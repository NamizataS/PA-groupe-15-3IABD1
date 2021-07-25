import os
import random
import numpy as np
from PIL import Image
import json

from matplotlib import pyplot as plt

from sklearn_wrapper import MySKLearnMLPRawWrapper
import tqdm
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

path_to_shared_library = "target/release/librust_lib.dylib"
def toList(arr):
    return [j for i in arr for j in i]
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
                    image = image.resize((32, 32))
                    im_arr = np.array(image).flatten()
                    new_row = {'image': file_path, 'genre': lineJson['genre']}
                    dataset.append(new_row)


def import_dataset(file_path, folder):
    dataset = []
    fill_x_and_y(file_path, folder, dataset)
    return dataset


if __name__ == '__main__':
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])
    d = [2, 2, 1]

    dataset_inputs = X
    dataset_expected_outputs = Y

    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-5, 15, 2) for x2 in range(-5, 15, 2)]
    colors = ["blue" if output >= 0 else "red" for output in dataset_expected_outputs]
    wrapped_model = MySKLearnMLPRawWrapper(d, iteration_count=100)
    losses = []

    for epoch in tqdm.tqdm(range(1000)):
        wrapped_model.fit(dataset_inputs, dataset_expected_outputs)
        predicted_y_train = wrapped_model.predict(dataset_inputs, 2, 2)
        predicted_y_test = wrapped_model.predict(test_dataset, 2, 2)

        loss = mean_squared_error(dataset_expected_outputs, predicted_y_train)
        losses.append(loss)

        plt.plot(losses)
        plt.legend('loss', loc='upper left')
        plt.title('Evolution of loss for MLP with Rust library')
        plt.xlabel('epochs')
        plt.ylabel('mean squared error')
        plt.show()

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

    with open("nouveau_dataset/train_dataset2.json", "w") as train_dataset_file:
        json.dump(train_dataset, train_dataset_file)
    with open('nouveau_dataset/test_dataset2.json', "w") as test_dataset_file:
        json.dump(test_dataset, test_dataset_file)