#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:47:22 2017

@author: sophie

Explained variance of the PCA step: 99%
(861, 300)
(861, 27)
loss = 3.58562
loss = 0.0568236
loss = 0.00908288
loss = 0.00378182
loss = 0.00209687
loss = 0.00133569
loss = 0.000926758
loss = 0.000680892
loss = 0.000520877
loss = 0.00041071
1.0
0.99422

"""

import scipy.io as sio
import glob
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


NUM_CATEGORIES = 27

def load_input(path, key):
    X = []
    y = []
    filenames = glob.glob(os.path.join(path, '*.mat'))
    for filename in filenames:
        X.append(sio.loadmat(filename)[key])
        y.append(one_hot(int(filename.split('_')[0][len(path) + 2:])))
    return X, np.array(y)

def preprocess(X, y):
    min_frame = min(X, key=lambda x: x.shape[2]).shape[2]
    X = np.array([x[:,:,:min_frame].flatten() for x in X])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    pca = PCA(300)
    X = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_.sum()
    print("Explained variance of the PCA step: {}%".format(int(explained_variance * 100)))
    return np.array(X), np.array(y)

def one_hot(i):
    b = np.zeros(NUM_CATEGORIES)
    b[i-1] = 1
    return b