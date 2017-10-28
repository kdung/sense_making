#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 17:44:54 2017

@author: sophie
"""
import numpy as np

def nn(X, y, nodes = 3, alpha = 0.01, iter = 1000):
    num_features = 3
    w1 = np.random.random_sample((nodes, num_features))
    w2 = np.random.random_sample((2, nodes))
    print(w1)
    print(w2)
    print(X[0])
    for i in range(iter):
        z1 = np.dot(X, w1.T) # m x nodes
        
        a1 = sigmoid(z1) # m x nodes
        h = np.dot(a1, w2.T) # m x 2
        if i % 100 == 0:
            print("-----------------------------------")
            print("h ----")
            print(h[0])
            print("y-----")
            print(y[0])
        m = X.shape[0]
        J =  (1/(2 * m)) * np.sum(np.square(h - y))
        if i % 100 == 0:
            print("cost = %s " % J)
        
        dh = d_cost(y, h)
        da1 = w2
        dX = w1
        dz1 = d_reLU(a1)
        # dX = np.dot(z1, dz1) # m x 3
        delta = np.dot( d_sigmoid(np.dot(dh, da1)), dX)/m
        if i % 100 == 0:
            print("Z1")
            print(z1[0])
        if i % 100 == 0:
            print("X")
            print(X[0])
    
        if i % 100 == 0:
            print("delta")
            print(delta[0])
        
        X = X - delta * alpha
         

def sigmoid(x):
    z = np.exp(-x)
    return z / (1 + z)

def reLU(x):
    return np.maximum(x, 0)

def d_reLU(x):
    return 1 * (x > 0)
 
def d_cost(y, h):
    return h - y

def d_sigmoid(x):
    return x * (1 - x)

def normalize(X):
    return (X - np.mean(X, axis = 1, keepdims = True))/(np.max(X, axis = 1, keepdims = True)
            - np.min(X, axis = 1, keepdims = True))
    

joint_count = 3
joints = np.random.random_sample((1,3))

thetas = np.pi * 2 * np.random.random_sample((1000, 3)) - np.pi
print("theta shape = %s" % str(thetas.shape))
thetas_sum = np.array([thetas[:,0], 
              thetas[:,0] + thetas[:,1],
              thetas[:,0] + thetas[:,1] + thetas[:,2]])
cos_thetas = np.cos(thetas_sum)

x_partials = cos_thetas.T * joints

x = np.sum(x_partials, axis = 1, keepdims = True)
print("x shape = %s" % str(x.shape))
sin_thetas = np.sin(thetas_sum)
y_partials = sin_thetas.T * joints
y = np.sum(y_partials, axis = 1, keepdims = True)
print("y shape = %s" % str(y.shape))

output = np.concatenate((x, y), axis = 1)
print("output shape = %s" % str(output.shape))
nn(normalize(thetas), output)
