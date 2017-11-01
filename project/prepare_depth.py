#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 23:16:04 2017

@author: sophie
"""

import scipy.io as sio
import glob
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.random_projection import SparseRandomProjection

NUM_CATEGORIES = 27
"""
(96, 300)
(96, 3)
loss = 15.0341
loss = 2.85778e-05
loss = 1.16586e-05
loss = 7.08497e-06
loss = 4.91575e-06
loss = 3.68604e-06
loss = 2.83434e-06
loss = 2.12536e-06
loss = 1.705e-06
loss = 1.41482e-06
1.0
0.8

(861, 300)
(861, 27)
loss = 41.9578
loss = 0.0466252
loss = 0.00952245
loss = 0.00452041
loss = 0.00266922
loss = 0.00179631
loss = 0.00129072
loss = 0.000976606
loss = 0.000765749
loss = 0.000617511
1.0
0.693642
"""

def load_input(path, key):
    X = []
    y = []
    filenames = glob.glob(os.path.join(path, '*.mat'))
    
    for index, filename in enumerate(filenames):
        X.append(sio.loadmat(filename)[key])
        y.append(one_hot(int(filename.split('_')[0][len(path) + 2:])))
    return X, np.array(y)

def preprocess(X, y):
    min_frame = min(X, key=lambda x: x.shape[2]).shape[2]
    X = np.array([x[:,:,:min_frame].flatten() for x in X])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    #sp = SparseRandomProjection(n_components = 300)
    #X_transform = sp.fit_transform(X)
    return np.array(X), np.array(y)

def one_hot(i):
    b = np.zeros(NUM_CATEGORIES)
    b[i-1] = 1
    return b

if __name__ == "__main__":
    X, y = load_input('test', 'd_depth')
    X, y = preprocess(X, y)
    print(X.shape)
