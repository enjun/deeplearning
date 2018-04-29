# matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
    plt.show()
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)
# uint不能有负数，我们先转为float类型
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = (X_train - 127) / 127
X_test = (X_test - 127) / 127
nb_classes = 10
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()

model.add(Dense(512, input_shape=(784,),
                kernel_initializer=keras.initializers.TruncatedNormal(mean=.0, stddev=.05, seed=None)))
model.add(Activation('relu'))
model.add(Dropout(0))

# model.add(Dense(512, kernel_initializer=keras.initializers.TruncatedNormal(mean=.0, stddev=.05, seed=None)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, batch_size=64, verbose=1, validation_split=0.05)
# validation_split=0.05: take the last 5% of the train data as validation data, \
# (Note that, this operation is before shuttle, so first shuttle train data then split it)
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Accuracy:', accuracy)
