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
from sklearn.random_projection import johnson_lindenstrauss_min_dim

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

(861, 1000)
(861, 27)
loss = 16.246
loss = 0.0145152
loss = 0.00424126
loss = 0.00206465
loss = 0.00117178
loss = 0.000727882
loss = 0.000501082
loss = 0.000369592
loss = 0.000282762
loss = 0.000220575
1.0
0.82659

(861, 5792)
(861, 27)
loss = 7.6444
loss = 0.0484374
loss = 0.0141503
loss = 0.00717282
loss = 0.00435913
loss = 0.00289571
loss = 0.00206019
loss = 0.00152134
loss = 0.00116923
loss = 0.000930001
1.0
0.965318

from sklearn.random_projection import johnson_lindenstrauss_min_dim
johnson_lindenstrauss_min_dim(861,eps=0.1)
Out[3]: 5792
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
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    
    # reduce principle components to improve performance
    reduced_pc = 2000
    recommended_pc = johnson_lindenstrauss_min_dim(861,eps=0.1)
    min_pc = recommended_pc - reduced_pc
    sp = SparseRandomProjection(n_components = int(min_pc))
    X = sp.fit_transform(X)
    return np.array(X), np.array(y)

def one_hot(i):
    b = np.zeros(NUM_CATEGORIES)
    b[i-1] = 1
    return b

if __name__ == "__main__":
    X, y = load_input('test', 'd_depth')
    X, y = preprocess(X, y)
    print(X.shape)
