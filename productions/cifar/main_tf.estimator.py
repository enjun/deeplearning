from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import os
from functools import reduce
import readcifar

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate')
tf.app.flags.DEFINE_string('model_folder', 'model', 'folder where to storage model')
tf.app.flags.DEFINE_integer('epochs', 2, 'times of epochs for training')
tf.app.flags.DEFINE_integer('batch_size', 128, 'number of a batch for training and testing')
FLAGS = tf.app.flags.FLAGS


def normalize_data(*kw):
    result = []
    for data in kw:
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std
        result.append(data)
    return result


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 32, 32, 3], name='input_layer_cnn')
    tf.summary.image('input', input_layer)
    conv1 = tf.layers.conv2d(input_layer, 64, [3, 3],
                             strides=[2, 2],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(),
                             name='conv1')
    pooling1 = tf.layers.max_pooling2d(conv1, [2, 2], strides=[2, 2], name='max_pooling1')
    tf.summary.scalar('p1', tf.reduce_mean(pooling1))
    conv2 = tf.layers.conv2d(pooling1, 128, [2, 2],
                             strides=[1, 1],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(),
                             name='conv2')
    pooling2 = tf.layers.max_pooling2d(conv2, [2, 2], strides=[1, 1], name='max_pooling2')
    tf.summary.scalar('p2', tf.reduce_mean(pooling2))
    conv3 = tf.layers.conv2d(pooling2, 256, [2, 2],
                             strides=[1, 1],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(),
                             name='conv3')
    tf.summary.histogram('conv3', tf.reduce_mean(conv3))
    # flatten conv3
    num = reduce(lambda a, b: a * b, conv3.get_shape().as_list()[1:4])
    flatten_conv3 = tf.reshape(conv3, shape=[-1, num])

    dense1 = tf.layers.dense(flatten_conv3, 350,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(),
                             name='dense1')

    dense2 = tf.layers.dense(dense1, 350,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(),
                             name='dense2')

    dropout1 = tf.layers.dropout(dense2, rate=0.2,
                                 training=mode == tf.estimator.ModeKeys.TRAIN,
                                 name='dropout1')

    logits = tf.layers.dense(dropout1, 10,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(),
                             name='logits')

    predictions = {'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
                   'probabilities': tf.nn.relu(logits, name='prediction_prob')}#,,,,,,,,,,

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    summary_hook = tf.train.SummarySaverHook(save_steps=10,
                                             output_dir='model',
                                             summary_op=tf.summary.merge_all())

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.003, momentum=0.8, name='RMSprop')
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])

    eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(args):
    if not os.path.exists(FLAGS.model_folder):
        os.mkdir(FLAGS.model_folder)
    # first load dataset and normalize data
    # reshape image style into [samples, width, height, channels]
    if not os.path.exists(r'dir_data'):
        os.mkdir(r'dir_data')
        (x_train, y_train), (x_test, y_test), label_names = readcifar.read_cifar(r'../../dataset/cifar-10-batches-py')
        x_train, x_test = normalize_data(x_train, x_test)
        x_test = readcifar.to_image(x_test, 10000)
        x_train = readcifar.to_image(x_train, 50000)
        np.savez(r'dir_data/out.npz', x_train, y_train, x_test, y_test, label_names)

    r = np.load(r'dir_data/out.npz')
    x_train = r['arr_0'].astype(np.float32)
    y_train = r['arr_1'].astype(np.int32)
    x_test = r['arr_2'].astype(np.float32)
    y_test = r['arr_3'].astype(np.int32)
    label_names = r['arr_4']

    # split dataset into train dataset and validation dataset
    num = int(0.8 * len(x_train))
    x_tra, x_val = x_train[:num], x_train[num:]
    y_tra, y_val = y_train[:num], y_train[num:]

    tensors_to_log = {'probabilities': 'prediction_prob'}
    logging_hook = tf.train.LoggingTensorHook(tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x_test},
                                                        y=y_test,
                                                        batch_size=FLAGS.batch_size,
                                                        num_epochs=None,
                                                        shuffle=True)
    estimator_cifar10 = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=FLAGS.model_folder)
    estimator_cifar10.train(input_fn=train_input_fn, hooks=[logging_hook], steps=400)

    # evaluate trained model in test dataset
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x_test},
                                                       y=y_test,
                                                       batch_size=FLAGS.batch_size,
                                                       num_epochs=1,
                                                       shuffle=False)
    result_test = estimator_cifar10.evaluate(input_fn=eval_input_fn)
    print(result_test)


if __name__ == '__main__':
    tf.app.run()
