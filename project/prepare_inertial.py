#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:29:37 2017

@author: sophie
"""

import scipy.io as sio
import glob
import os
import numpy as np

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
    min_frame = X[0].shape[0]
    for  x in X:
        if min_frame > x.shape[0]:
            min_frame = x.shape[0]
    print(min_frame)
    X = np.array([x[:min_frame,:].flatten() for x in X])
    
    n = X[0].shape[0]
    m = len(X)
    X = np.array([np.array(x.reshape(n)) for x in X])
    X = np.array(X.reshape(m,n))
    y = y.reshape(m, NUM_CATEGORIES)
    
    return np.array(X), y

def one_hot(i):
    b = np.zeros(NUM_CATEGORIES)
    b[i-1] = 1
    return b
