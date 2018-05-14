import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re


def read_cifar(dir):
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
    quotient = num // 1000
    remainder= num % 1000
    for i in range(quotient):
        temp_x = np.array([[[a[i], a[i + 1024], a[i + 2048]] for i in range(1024)] for a in data[i*1000:(i+1)*1000]])
        temp_x = np.reshape(temp_x, [-1, 32, 32, 3])
        if i == 0:
            images_x = temp_x
        else:
            images_x = np.concatenate([images_x, temp_x], axis=0)
    if remainder != 0:
        temp_x = np.array([[[a[i], a[i + 1024], a[i + 2048]] for i in range(1024)] for a in
                           data[quotient * 1000:quotient * 1000 + remainder]])
        temp_x = np.reshape(temp_x, [-1, 32, 32, 3])
        if quotient != 0:
            images_x = np.concatenate([images_x, temp_x], axis=0)
        else:
            images_x = temp_x
    return images_x


def show_images(images):
    for i in range(10):
        plt.imshow(images[i + 20])
        plt.show()


# dir = r'../../dataset/cifar-10-batches-py'
# (x_train, y_train), (x_test, y_test), label_names = read_cifar(dir)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(label_names)



