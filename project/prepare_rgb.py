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



NUM_CATEGORIES = 27
RESIZED_W = 200
RESIZED_H = 200
SKIP_NUMBER = 1

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
        cv2.destroyAllWindows()
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
        #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0
        while(cap.isOpened()):
            _, frame = cap.read()
            if frame is None:
                break
            frame_idx += 1
        fr_dict[filename] = frame_idx
        
        
        #print(X.shape) (21, 200, 200, 3)
        cap.release()
        cv2.destroyAllWindows()
    
    np.save(output, fr_dict)
    
    return fr_dict

def clean_data(input_path, out_path, frames = 'rgb_frame.npy', frame_dict = 'rgb_fr_dict.npy'):
    filenames = glob.glob(os.path.join(input_path, '*.avi'))
    frames = np.load(frames)
    min_fr = min(frames)
    
    fr_dict = np.load(frame_dict)
    file_dict = {}
    for index, filename in enumerate(filenames):
        X = []
        cap = cv2.VideoCapture(filename)
        #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        clazz = filename.split('_')[0][len(input_path) + 2:]
        frame_idx = 0
        skip_no = int(fr_dict[str(filename)]/min_fr)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            frame_idx += 1
            if frame_idx % skip_no != 0 or frame_idx > min_fr:
                continue
            frame = cv2.resize(frame, (RESIZED_W, RESIZED_H))
            
            X.append(frame)
        X = np.array(X)
        # new output npy: class_file-index_num-of-frames
        np_file_name = out_path + '/' + clazz + '_' + str(index) + '_' + str(X.shape[0])
        file_dict[np_file_name] = index
        np.save(np_file_name, X)
        
        #print(frame_idx)
        #print(X.shape) (21, 200, 200, 3)
        cap.release()
        cv2.destroyAllWindows()
        
def preprocess(X, y):
    min_frame = min(X, key=lambda x: x.shape[0]).shape[0]
    X = np.array([x[:min_frame,:,:,:].flatten() for x in X])
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
    
    clean_data('RGB', 'rgb_clean')
    #total_frames = get_no_frames('RGB', 'rgb_npy')
    #fr_dict = get_no_frames_dict('RGB', 'rgb_fr_dict')
    