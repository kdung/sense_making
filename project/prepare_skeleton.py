#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:47:22 2017

@author: sophie
"""

import scipy.io as sio
import glob
import os
import numpy as np

NUM_CATEGORIES = 27

def load_data(path, dict_key, N):
    filenames = glob.glob(os.path.join(path, '*.mat'))

    X = []
    y = []

    M = 0

    for filename in filenames:
        clss = int(filename.split('_')[0][len(path) + 2:]) - 1
        if clss < NUM_CATEGORIES:
            x = sio.loadmat(filename)[dict_key][:,:,:41]
            X.append(x.reshape(N))
            y.append(one_hot(clss))
            M += 1

    return np.array(X), np.array(y)

def one_hot(i):
    b = np.zeros(NUM_CATEGORIES)
    b[i] = 1
    return b
