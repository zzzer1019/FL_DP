# ==============================
# Builds the MNIST network.
# ==============================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.stats import truncnorm

import numpy as np
import math
import tensorflow as tf

#the digits 0 through 9
NUM_CLASSES = 10

#28x28
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def init_weights(size):

    return np.float32(truncnorm.rvs(-2, 2, size=size)*1.0/math.sqrt(float(size[0])))


def inference(images, Hidden1, Hidden2, Hidden3):

  with tf.name_scope('hidden1'):

    weights = tf.Variable(init_weights([IMAGE_PIXELS, Hidden1]), name='weights', dtype=tf.float32)
    biases = tf.Variable(np.zeros([Hidden1]),name='biases',dtype=tf.float32)
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

  with tf.name_scope('hidden2'):

    weights = tf.Variable(init_weights([Hidden1, Hidden2]),name='weights',dtype=tf.float32)
    biases = tf.Variable(np.zeros([Hidden2]),name='biases',dtype=tf.float32)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  with tf.name_scope('hidden3'):

    weights = tf.Variable(init_weights([Hidden2, Hidden3]),name='weights',dtype=tf.float32)
    biases = tf.Variable(np.zeros([Hidden3]),name='biases',dtype=tf.float32)
    hidden2 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)

  with tf.name_scope('out'):

    weights = tf.Variable(init_weights([Hidden3, NUM_CLASSES]), name='weights',dtype=tf.float32)
    biases = tf.Variable(np.zeros([NUM_CLASSES]), name='biases',dtype=tf.float32)
    logits = tf.matmul(hidden2, weights) + biases

  return logits


def inference_no_bias(images, Hidden1, Hidden2):

  with tf.name_scope('hidden1'):

    weights = tf.Variable(init_weights([IMAGE_PIXELS, Hidden1]), name='weights', dtype=tf.float32)
    hidden1 = tf.nn.relu(tf.matmul(images, weights))

  with tf.name_scope('hidden2'):

    weights = tf.Variable(init_weights([Hidden1, Hidden2]),name='weights',dtype=tf.float32)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights))

  with tf.name_scope('out'):

    weights = tf.Variable(init_weights([Hidden2, NUM_CLASSES]), name='weights',dtype=tf.float32)
    logits = tf.matmul(hidden2, weights)

  return logits


#Calculates the loss from the logits and the labels.
def loss(logits, labels):

  labels = tf.to_int64(labels)

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')

  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):

  tf.summary.scalar('loss', loss)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  global_step = tf.Variable(0, name='global_step', trainable=False)

  train_op = optimizer.minimize(loss, global_step=global_step)

  return train_op


def evaluation(logits, labels):

  correct = tf.nn.in_top_k(logits, labels, 1)

  return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size):

    images_placeholder = tf.placeholder(tf.float32, shape=(None,IMAGE_PIXELS), name='images_placeholder')

    labels_placeholder = tf.placeholder(tf.int32, shape=(None), name='labels_placeholder')

    return images_placeholder, labels_placeholder


def mnist_cnn_model(batch_size):

    # - placeholder for the input Data (in our case MNIST), depends on the batch size specified in C
    data_placeholder, labels_placeholder = placeholder_inputs(batch_size)

    # Input Layer
    input_layer = tf.reshape(data_placeholder, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)

    # Fully connected Layer
    conv3_flat = tf.reshape(conv3, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=conv3_flat, units=600, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels_placeholder, logits=logits)

    eval_correct = evaluation(logits, labels_placeholder)

    global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    train_op = optimizer.minimize(loss=loss, global_step=global_step)

    return train_op, eval_correct, loss, data_placeholder, labels_placeholder


def mnist_fully_connected_model(batch_size, hidden1, hidden2):

    data_placeholder, labels_placeholder = placeholder_inputs(batch_size)

    logits = inference_no_bias(data_placeholder, hidden1, hidden2)

    Loss = loss(logits, labels_placeholder)

    eval_correct = evaluation(logits, labels_placeholder)

    global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')

    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=27000,                                                                       decay_rate=0.1, staircase=False, name='learning_rate')

    train_op = training(Loss, learning_rate)

    return train_op, eval_correct, Loss, data_placeholder, labels_placeholder
