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
from sklearn.decomposition import TruncatedSVD

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
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    svd = TruncatedSVD(300)
    X_lsa = svd.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    return np.array(X_lsa), np.array(y)

def one_hot(i):
    b = np.zeros(NUM_CATEGORIES)
    b[i-1] = 1
    return b

if __name__ == "__main__":
    X, y = load_input('test', 'd_depth')
    preprocess(X, y)
