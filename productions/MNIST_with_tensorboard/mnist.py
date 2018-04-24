# CNN to recognize the figures in the mnist dataset
# start using tensorboard to adjusting model

import tensorflow as tf
import numpy as np
import os
from functools import reduce
from tensorflow.python.framework import ops
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

ops.reset_default_graph()


def conv_layer(input, channels_in, channels_out, name='conv_layer'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(shape=[4, 4, channels_in, channels_out], stddev=0.1, dtype=tf.float32), name='conv_w')
        b = tf.Variable(tf.constant(0.1, shape=[channels_out], dtype=tf.float32), name='conv_b')
        out = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME', name='conv_out')
        tf.summary.histogram('W', W)
        tf.summary.histogram('b', b)
        return tf.nn.relu(out + b)


def fc_layer(input, channels_in, channels_out, name='fc_layer'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([channels_in, channels_out], dtype=tf.float32), name='fc_w')
        b = tf.Variable(tf.constant(0.1, shape=[channels_out], dtype=tf.float32), name='fc_b')
        tf.summary.histogram('W', W)
        tf.summary.histogram('b', b)
        out = tf.add(tf.matmul(input, W), b)
        return tf.nn.relu(out)


def main(args):
    with tf.Graph().as_default():
        sess = tf.Session()
        # add some directory in current working directory
        dir_tb = 'tensorboard2'
        if not os.path.exists(dir_tb):
            os.mkdir(dir_tb)
        dir_model = 'train_model1'
        if not os.path.exists(dir_model):
            os.mkdir(dir_model)


        # setting some parameters
        epochs = 200
        batch_size = 100

        mnist = read_data_sets('temp')
        # Convert images into 28x28 (they are downloaded as 1x784)
        x_train = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
        x_test = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])

        # Convert labels into one-hot encoded vectors
        y_train = mnist.train.labels
        y_test = mnist.test.labels

        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        with tf.name_scope('placehoder'):
            x_data = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='x_data')
            y_data = tf.placeholder(dtype=tf.int32, shape=[None], name='y_data')
            tf.summary.image('x', x_data)

        # convolutional neural network
        conv1 = conv_layer(x_data, 1, 32)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = conv_layer(pool1, 32, 64)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        num = reduce(lambda a,b: a*b, pool2.get_shape().as_list()[1:4])
        flatten_pool2 = tf.reshape(pool2, shape=[-1, num])
        fc1 = fc_layer(flatten_pool2, num, 128)
        fc2 = fc_layer(fc1, 128, 10)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2, labels=y_data), name='loss')
            tf.summary.scalar('loss', loss)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.0005).minimize(loss, name='optimizer')
        with tf.name_scope('accuracy'):
            equals = tf.equal(tf.argmax(fc2, axis=1), tf.cast(y_data, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(equals, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', accuracy)

        summary_all = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())

        writer_summary = tf.summary.FileWriter(dir_tb, sess.graph)

        for i in range(epochs):
            indices = np.random.choice(len(y_train), batch_size, replace=False)

            feed_dict = {x_data: x_train[indices].reshape([batch_size, 28, 28, 1]), y_data: y_train[indices]}
            _, summary_merge = sess.run([optimizer, summary_all], feed_dict=feed_dict)

            writer_summary.add_summary(summary_merge, i)
            if i % 10 == 0:
                loss_temp = sess.run(loss, feed_dict=feed_dict)
                accuracy_temp = sess.run(accuracy, feed_dict=feed_dict)
                print('Loss of epoch{}: {:.4}'.format(i+1, loss_temp))
                print('Accuracy of epoch{}: {:.4}%'.format(i + 1, accuracy_temp*100))



if __name__ == '__main__':
    tf.app.run()