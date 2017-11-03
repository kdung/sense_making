#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 23:27:59 2017

@author: sophie

https://pythonprogramming.net/loading-video-python-opencv-tutorial/
(640 480 3) * 42 frames
"""
import cv2
import glob
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import pickle
from generator import MiniBatchGenerator
from models import MLP
import tensorflow as tf
import models


NUM_CATEGORIES = 27
RESIZED_W = 200
RESIZED_H = 200
SKIP_NUMBER = 1
file_dict = {}
reduced_pc = 2000
recommended_pc = johnson_lindenstrauss_min_dim(861,eps=0.1) #5792
min_pc = recommended_pc - reduced_pc #3792

def get_no_frames(input_path, output):
    total_frames = []
    filenames = glob.glob(os.path.join(input_path, '*.avi'))
    for index, filename in enumerate(filenames):
        cap = cv2.VideoCapture(filename)
        #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0
        while(cap.isOpened()):
            _, frame = cap.read()
            if frame is None:
                break
            frame_idx += 1
        total_frames.append(frame_idx)
        
        #print(X.shape) (21, 200, 200, 3)
        cap.release()
    print(total_frames)
    print(min(total_frames)) #32
    print(max(total_frames)) #96
    np.save(output, total_frames)
    import matplotlib.pyplot as plt
    plt.hist(total_frames, normed=False, bins=30)
    plt.ylabel('rgb frames');
    return total_frames

def get_no_frames_dict(input_path, output):
    fr_dict = {}
    filenames = glob.glob(os.path.join(input_path, '*.avi'))
    for index, filename in enumerate(filenames):
        cap = cv2.VideoCapture(filename)
        frame_idx = 0
        while(cap.isOpened()):
            _, frame = cap.read()
            if frame is None:
                break
            frame_idx += 1
        fr_dict[filename] = frame_idx
        #print(X.shape) (21, 200, 200, 3)
        cap.release()
    np.save(output, fr_dict)
    return fr_dict

def clean_data(input_path, out_path, frames = 'rgb_frame.npy', frame_dict = 'rgb_fr_dict.npy'):
    filenames = glob.glob(os.path.join(input_path, '*.avi'))
    frames = np.load(frames)
    min_fr = min(frames)
    
    fr_dict = np.load(frame_dict).tolist()
    file_dict = {}
    for index, filename in enumerate(filenames):
        X = []
        cap = cv2.VideoCapture(filename)
        clazz = filename.split('_')[0][len(input_path) + 2:]
        frame_idx = 0
        skip_no = int(fr_dict[str(filename)]/min_fr)
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            frame_idx += 1
            if frame_idx % skip_no != 0 or len(X) >= min_fr:
                continue
            frame = cv2.resize(frame, (RESIZED_W, RESIZED_H))
            
            X.append(frame)
        X = np.array(X)
        # new output npy: class_file-index_num-of-frames
        np_file_name = out_path + '/' + clazz + '_' + str(index) + '_' + str(X.shape[0])
        file_dict[index] = np_file_name
        np.save(np_file_name, X)
        
        #print(frame_idx)
        #print(X.shape) (21, 200, 200, 3)
        cap.release()
    with open('file_dict.pickle', 'wb') as handle:
        pickle.dump(file_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return file_dict


with open('file_dict.pickle', 'rb') as handle:
        file_dict = pickle.load(handle)
    
def x_mapper(item):
    return np.load(file_dict[item] + ".npy")

def y_mapper(item):
    return int(file_dict[item].split('/')[1].split('_')[0])

def preprocess(X, y):
    X = np.array([x.flatten() for x in X])
    y = np.array([one_hot(y_item) for y_item in y])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    #print(X.shape)  #(173, 3840000)
    # reduce principle components to improve performance
    
    sp = SparseRandomProjection(n_components = int(5792))
    X = sp.fit_transform(X)
    return np.array(X), y

def one_hot(i):
    b = np.zeros(NUM_CATEGORIES)
    b[i-1] = 1
    return b

def preprocess_batch():
    generator = MiniBatchGenerator(861, x_mapper, y_mapper)
    generator.split_train_test()
    print('load train mini-batch')
    X_all = []
    y_all = []
    while True:
        X, y = generator.load_next_train_batch(100)
        if X is None:
            break
        X, y = preprocess(X, y)
        X_all.extend(X)
        y_all.extend(y)
    
    with open('rgb_resized/X_train.pickle', 'wb') as handle:
        pickle.dump(X_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('rgb_resized/y_train.pickle', 'wb') as handle:
        pickle.dump(y_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    X_test, y_test = generator.load_next_test_batch(1000)
    X_test, y_test = preprocess(X_test, y_test)
    
    with open('rgb_resized/X_test.pickle', 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('rgb_resized/y_test.pickle', 'wb') as handle:
        pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def train_minibatch():

    generator = MiniBatchGenerator(861, x_mapper, y_mapper)
    generator.split_train_test()
    
    """
    print('load train mini-batch')
    while True:
        X, y = generator.load_next_train_batch(10)
        if X is None:
            break
        print(y)
        X_to_train, y_to_train = preprocess(X, y)
        print(X_to_train.shape)
        print(y_to_train.shape)
        break
        
        sess, y_predict, x_train, y_train = models.train_nn(NUM_CATEGORIES,
                                                        X_to_train, y_to_train,
                                                        layers=(256, 256),
                                                        iterations=1000)

    print('load test mini-batch')
    while True:
        X, y = generator.load_next_test_batch(10)
        if X is None:
            break
        X_to_test, y_to_test = preprocess(X, y)
        correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_train, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x_train: X_to_test, y_train: y_to_test}))
        
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #print(sess.run(accuracy, feed_dict={x_train: X_to_train, y_train: y_to_train}))
    print(sess.run(accuracy, feed_dict={x_train: X_to_test, y_train: y_to_test}))
    """
    
    model = MLP(NUM_CATEGORIES, min_pc, layers=(256, 256, 256), learning_rate=0.0005)
    model.start_session()
    
    X_test, y_test = generator.load_next_test_batch(1000)
    X_test, y_test = preprocess(X_test, y_test)
    #sess = None
    for iteration in range(1000):
        generator.reset()
        while True:
            X_train, y_train = generator.load_next_train_batch(10)
            if X_train is None:
                break
            X_train, y_train = preprocess(X_train, y_train)
            model.train(X_train, y_train)
        if iteration % 100 == 0:
            print('loss = ', model.calculate_loss())
            # print('train acc = ', model.calculate_accuracy(X_train, y_train))
            print('test acc = ', model.calculate_accuracy(X_test, y_test))
            print("\n---\n")
    #saver = tf.train.Saver()
    #save_path = saver.save(sess, "/model/rgb.ckpt")
    #print("Model saved in file: %s" % save_path)
    
    #print(model.calculate_accuracy(X_test, y_test))
    
def train():
    with open('rgb_resized/X_train.pickle', 'rb') as handle:
        X_to_train = np.array(pickle.load(handle))
    with open('rgb_resized/X_test.pickle', 'rb') as handle:
        X_to_test = np.array(pickle.load(handle))
    with open('rgb_resized/y_train.pickle', 'rb') as handle:
        y_to_train = np.array(pickle.load(handle))
    with open('rgb_resized/y_test.pickle', 'rb') as handle:
        y_to_test = np.array(pickle.load(handle))
        
    print(X_to_train.shape)
    print(y_to_train.shape)

    sess, y_predict, x_train, y_train = models.train_nn(NUM_CATEGORIES,
                                                        X_to_train, y_to_train,
                                                        layers=(256, 256),
                                                        iterations=1000)

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x_train: X_to_train, y_train: y_to_train}))
    print(sess.run(accuracy, feed_dict={x_train: X_to_test, y_train: y_to_test}))
    
if __name__ == "__main__":
    
    #clean_data('RGB', 'rgb_clean')
    #total_frames = get_no_frames('RGB', 'rgb_npy')
    #fr_dict = get_no_frames_dict('RGB', 'rgb_fr_dict')
    #preprocess_batch()
    train()
    