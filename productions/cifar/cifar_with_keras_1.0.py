"""
This program can classify ten different things in dataset of cifar10
Version: 1.0
Date: 2018/05/13
author: Enjun Xia
"""
from __future__ import print_function

import numpy as np
import tensorboard as tf
import os

import keras
from keras.models import Sequential, Model
from keras.layers.core import *
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from readcifar import read_cifar, to_image  # it is my personal module


def mymodel():
    model1 = Sequential()

    model1.add(Conv2D(32, [3, 3], strides=(1, 1), padding='same',
                      kernel_initializer=keras.initializers.TruncatedNormal(),
                      kernel_regularizer='l2', input_shape=[32, 32, 3], name='conv1'))
    # model1.add(MaxPooling2D([2, 2], strides=[2, 2]))
    model1.add(Activation('relu'))

    model1.add(Conv2D(64, [2, 2], strides=(2, 2),
                      kernel_initializer=keras.initializers.TruncatedNormal(),
                      kernel_regularizer='l2', name='conv2'))
    model1.add(MaxPooling2D([2, 2], strides=[1, 1]))
    model1.add(Activation('relu'))

    model1.add(Conv2D(128, [2, 2], strides=(1, 1),
                      kernel_initializer=keras.initializers.TruncatedNormal(),
                      kernel_regularizer='l2', name='conv3'))
    model1.add(MaxPooling2D([2, 2], strides=[1, 1]))
    model1.add(Activation('relu'))

    # model1.add(Conv2D(256, [2, 2], strides=(1, 1),
    #                   kernel_initializer=keras.initializers.TruncatedNormal(),
    #                   kernel_regularizer='l2', name='conv4'))
    # # model1.add(MaxPooling2D([2, 2], strides=[1, 1]))
    # model1.add(Activation('relu'))

    model1.add(Flatten())
    model1.add(Dense(256, activation='relu'))
    model1.add(Dense(256, activation='relu'))
    # model1.add(Dropout(0.3))
    model1.add(Dense(10, activation='softmax'))
    return model1


def main():
    version = 'cifar_with_keras_1.0'
    dir = r'../../dataset/cifar-10-batches-py'
    if not os.path.exists(r'model'):
        os.mkdir(r'model')

    (x_trains, y_trains), (x_test, y_test), label_names = read_cifar(dir)
    y_trains = np_utils.to_categorical(y_trains, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    batch_size = 128
    epochs = 40
    if not os.path.exists(r'dir_data'):
        os.mkdir(r'dir_data')
        x_test = to_image(x_test, 10000)
        x_trains = to_image(x_trains, 50000)
        np.savez(r'dir_data/out.npz', x_trains, x_test)

    r = np.load(r'dir_data/out.npz')
    x_trains = r['arr_0']
    x_test = r['arr_1']

    # split data into train dataset and validation dataset
    num_temp = int(0.8 * len(x_trains))
    x_train, y_train = x_trains[:num_temp], y_trains[:num_temp]
    x_val, y_val = x_trains[num_temp:], y_trains[num_temp:]

    if os.path.exists(r'model/model_{}.h5'.format(version)):
        m = keras.models.load_model(r'model/model_{}.h5'.format(version))
    else:
        m = mymodel()

    datagen_train = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    datagen_train.fit(x_train)
    datagen_val = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    datagen_val.fit(x_val)
    datagen_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    datagen_test.fit(x_test)

    m.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9), loss='categorical_crossentropy',
              metrics=['accuracy'])
    tb = keras.callbacks.TensorBoard('tensorboard', histogram_freq=0, batch_size=batch_size)
    save_model = keras.callbacks.ModelCheckpoint(r'model/model_{}.h5'.format(version), monitor='val_loss', verbose=1)
    rd = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

    history = m.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size,
                    epochs=epochs, validation_data=datagen_val.flow(x_val, y_val, batch_size=batch_size), verbose=2,
                    validation_steps=len(x_val) / batch_size, callbacks=[tb, save_model, rd])
    loss, accuracy = m.evaluate_generator(datagen_test.flow(x_test, y_test, batch_size=batch_size),
                                          steps=len(x_test) / batch_size)
    print('accuracy:{}  loss:{}'.format(accuracy, loss))
    keras.summ


if __name__ == '__main__':
    main()
