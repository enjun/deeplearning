# CNN to recognize the figures in the mnist dataset
# start using tensorboard to adjusting model

import tensorflow as tf
import numpy as np
import os
from functools import reduce


def conv_layer(input, channels_in, channels_out):
    # x = tf.reshape(input, [-1, 28, 28, 1])
    W = tf.Variable(tf.truncated_normal(shape=[5, 5, channels_in, channels_out]), dtype=tf.float32, name='conv_w')
    b = tf.Variable(tf.truncated_normal(shape=[channels_out]), dtype=tf.float32, name='conv_b')
    out = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME', name='conv_out')
    return tf.nn.relu(out + b)


def fc_layer(input, channels_in, channels_out):
    W = tf.Variable(tf.truncated_normal([channels_in, channels_out], dtype=tf.float32), name='fc_w')
    b = tf.Variable(tf.truncated_normal([channels_out], dtype=tf.float32), name='fc_b')
    out = tf.add(tf.matmul(input, W), b)
    return tf.nn.relu(out)


dir_tb = 'tensorboard'
if not os.path.exists(dir_tb):
    os.mkdir(dir_tb)
epochs = 200
batch_size = 40

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_data = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='x_data')
y_data = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_data')
# x_image = tf.reshape(x_data, [-1, 28, 28, 1])

conv1 = conv_layer(x_data, 1, 16)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv2 = conv_layer(pool1, 16, 32)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
flatten_pool2 = tf.reshape(pool2, shape=[-1, 7*7*32])
fc1 = fc_layer(flatten_pool2, 7*7*32, 128)
fc2 = fc_layer(fc1, 128, 1)

loss = tf.reduce_mean(tf.nn.l2_loss(fc2 - y_data), name='loss')
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss, name='optimizer')
equals = tf.equal(tf.argmax(fc2, axis=1), tf.cast(y_data, tf.int64))
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    indices = np.random.choice(len(y_train), batch_size, replace=False)
    feed_dict = {x_data: x_train[indices].reshape([batch_size, 28, 28, 1]), y_data: y_train[indices].reshape([batch_size, 1])}
    sess.run(optimizer, feed_dict=feed_dict)
    if i % 20 ==0:
        loss_temp = sess.run(loss, feed_dict=feed_dict)
        accuracy_temp = sess.run(accuracy, feed_dict=feed_dict)
        print('Loss of epoch{}: {:.4}'.format(i+1, loss_temp))
        print('Accuracy of epoch{}: {:.4}%'.format(i + 1, accuracy_temp*100))