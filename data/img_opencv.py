#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:28:33 2017

@author: sophie
ref: https://docs.opencv.org/3.2.0/d1/db7/tutorial_py_histogram_begins.html
"""

import cv2
from matplotlib import pyplot as plt

def show_hist(img_url):
    img = cv2.imread(img_url, 0)
    hist1 = cv2.calcHist([img],[0], None, [256], [0,256])

# hist,bins = np.histogram(img1.ravel(),256,[0,256])

    plt.hist(img.ravel(),256,[0,256])
    plt.show()
    
def show_hist_rgb(img_url):
    img1_full = cv2.imread(img_url)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img1_full],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()
    
img_url = 'data/hero1.jpg'
images_path = (
        ('hero1','data/hero1.jpg'), 
        ('hero2','data/hero2.jpg'), 
        ('hero3','data/hero3.jpg'), 
        ('hero4','data/hero4.jpg'))
OPENCV_METHODS = {
	"Correlation": cv2.HISTCMP_CORREL,
	"Chi-Squared": cv2.HISTCMP_CHISQR,
	"Intersection": cv2.HISTCMP_INTERSECT, 
	"Hellinger": cv2.HISTCMP_BHATTACHARYYA}

def get_hist(img_path):
    img = cv2.imread(img_path)
    return cv2.calcHist([img],[0,1,2], None, [256,256,256], [0,256,0,256,0,256])

#show_hist(img_url)
#show_hist_rgb(img_url)
def cal_similarity(images_path):
    hist_dict = {}
    for img, img_path in images_path:
        hist = get_hist(img_path)
        hist = cv2.normalize(hist, hist).flatten()
        hist_dict[img] = hist
    results = {}
    for (img, hist) in hist_dict.items():
        d = cv2.compareHist(hist_dict['hero1'], hist, OPENCV_METHODS["Intersection"])
        results[img] = d
    print(results)
    
cal_similarity(images_path)