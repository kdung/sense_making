#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:04:55 2017

@author: sophie
"""

import tensorflow as tf
from sklearn import model_selection as ms
import prepare_inertial
import prepare_skeleton
import prepare_depth
import models

NUM_CATEGORIES = 27

def train(X, y):
    print(X.shape)
    print(y.shape)
    X_to_train, X_to_test, y_to_train, y_to_test = ms.train_test_split(X, y, test_size=0.2, random_state=1)

    sess, y_predict, x_train, y_train = models.train_nn(NUM_CATEGORIES,
                                                        X_to_train, y_to_train,
                                                        layers=(256, 256),
                                                        iterations=1000)

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x_train: X_to_train, y_train: y_to_train}))
    print(sess.run(accuracy, feed_dict={x_train: X_to_test, y_train: y_to_test}))
    
def train_inertial():
    X, y = prepare_inertial.load_input('Inertial', 'd_iner')
    X, y = prepare_inertial.preprocess(X, y)
    train(X, y)

def train_skeleton():
    X, y = prepare_skeleton.load_input('Skeleton', 'd_skel')
    X, y = prepare_skeleton.preprocess(X, y)
    train(X, y)
    
def train_depth():
    X, y = prepare_depth.load_input('Depth', 'd_depth')
    X, y = prepare_depth.preprocess(X, y)
    train(X, y)
    
if __name__ == "__main__":
    #train_inertial()
    #train_skeleton()
    #train_depth()
    import cv2

    cap = cv2.VideoCapture('test/a1_s1_t1_color.avi')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
    
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(frame)
        #print(type(frame))
        if count > 1:
            break
        
        if frame is not None or count < 2:
            
            count += 1
        else:
            break
    print(count)
    cap.release()
    cv2.destroyAllWindows()
    
 
    