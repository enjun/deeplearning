import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data(path=r'D:\workspace\dataset\mnist\mnist.npz')

# uint不能有负数，我们先转为float类型
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = (X_train - 127) / 127
X_test = (X_test - 127) / 127
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print(X_train.shape)
nb_classes = 10
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model1 = Sequential()

model1.add(Conv2D(16, (2, 2), strides=(2, 2), padding='valid', activation=None, input_shape=(28, 28, 1)))
model1.add(MaxPooling2D((2, 2)))
model1.add(Activation('relu'))

model1.add(Flatten())
model1.add(Dense(10, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tb = keras.callbacks.TensorBoard(log_dir='temp',histogram_freq=1, write_graph=False)
model1.fit(X_train, y_train, batch_size=10, epochs=2, verbose=1, callbacks=[tb], validation_split=0.2)
loss, accuracy = model1.evaluate(X_test, y_test)
print('accuracy:{}'.format(accuracy))


