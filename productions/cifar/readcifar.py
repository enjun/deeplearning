import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re


def read_cifar(file):
    files_train = []
    for i in os.listdir(dir):
        if re.match(r'data.*', i):
            files_train.append(i)

    for i, file in enumerate(files_train):
        with open(os.path.join(dir, file), 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            temp_x = dict[b'data']
            temp_y = np.array(dict[b'labels'])
            if i == 0:
                x_train = temp_x
                y_train = temp_y
            else:
                x_train = np.vstack([x_train, temp_x])
                y_train = np.hstack([y_train, temp_y])

    with open(os.path.join(dir, 'test_batch'), 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        x_test = dict[b'data']
        y_test = np.array(dict[b'labels'])

    with open(os.path.join(dir, 'batches.meta'), 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        label_names = dict[b'label_names']

    return (x_train, y_train), (x_test, y_test), label_names


def to_image(data, num):
    images_x = np.array([[[a[i], a[i + 1024], a[i + 2048]] for i in range(1024)] for a in data[:num]])
    images_x = np.reshape(images_x, [-1, 32, 32, 3])
    return images_x


def show_images(images):
    for i in range(10):
        plt.imshow(images[i + 20])
        plt.show()


# dir = r'D:\workspace\dataset\cifar-10-batches-py'
# (x_train, y_train), (x_test, y_test), label_names = read_cifar(dir)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(label_names)



