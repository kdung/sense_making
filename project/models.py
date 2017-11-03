#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:31:54 2017

@author: sophie
"""
import tensorflow as tf

def train_nn(NUM_CATEGORIES, X_to_train, y_to_train, layers, iterations, learning_rate=0.0005):
    """train the model"""

    N = X_to_train.shape[1]

    # the input vector
    x_train = tf.placeholder(tf.float32, [None, N])

    # the ground truth vector
    y_train = tf.placeholder(tf.float32, [None, NUM_CATEGORIES])

    layer_i = x_train
    for i in layers:
        layer_i = tf.layers.dense(layer_i, i)
        layer_i = tf.nn.relu(layer_i)

    y_predict = tf.layers.dense(layer_i, NUM_CATEGORIES)

    # the loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_train))

    # the classifier
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # start the session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    # train with 1000 iterations
    for i in range(iterations):
        _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x_train: X_to_train, y_train: y_to_train})
        if i % 100 == 0:
            print('loss = ' + str(loss_val))

    return sess, y_predict, x_train, y_train

class MLP(object):

    def __init__(self, NUM_CATEGORIES, NUM_FEATURES, layers, learning_rate=0.0005):
        """train the model"""

        # the input vector
        x_train = tf.placeholder(tf.float32, [None, NUM_FEATURES])

        # the ground truth vector
        y_train = tf.placeholder(tf.float32, [None, NUM_CATEGORIES])

        layer_i = x_train
        for i in layers:
            layer_i = tf.layers.dense(layer_i, i)
            layer_i = tf.nn.relu(layer_i)

        y_predict = tf.layers.dense(layer_i, NUM_CATEGORIES)

        # the loss function
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_train))

        # the classifier
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        self.x_train = x_train
        self.y_train = y_train
        self.y_predict = y_predict
        self.cross_entropy = cross_entropy
        self.train_step = train_step
        self.sess = None
        self.loss_val = None

    def start_session(self):
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def train(self, X, y):
        _, self.loss_val = self.sess.run([self.train_step, self.cross_entropy], feed_dict={self.x_train: X, self.y_train: y})
        return self.sess

    def calculate_loss(self):
        return self.loss_val

    def calculate_accuracy(self, X, y):
        correct_prediction = tf.equal(tf.argmax(self.y_predict, 1), tf.argmax(self.y_train, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.sess.run(accuracy, feed_dict={self.x_train: X, self.y_train: y})
