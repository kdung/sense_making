#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:50:00 2017

"""

import random
import numpy as np

class MiniBatchGenerator(object):

    def __init__(self, max_size: int, x_mapper: callable, y_mapper: callable):
        self.arr = [idx for idx in range(max_size)]
        self.x_mapper = x_mapper
        self.y_mapper = y_mapper
        self.train_idx = 0
        self.train_size = max_size
        self.test_idx = max_size
        random.shuffle(self.arr)

    def split_train_test(self, test_size=0.2):
        self.test_idx = int(len(self.arr) * (1 - test_size))
        self.train_size = self.test_idx

    def load_next_train_batch(self, batch_size: int):
        batch = self.__next_batch(batch_size, self.train_idx, self.train_size)
        self.train_idx += len(batch)
        return self.__load(batch)

    def load_next_test_batch(self, batch_size: int):
        batch = self.__next_batch(batch_size, self.test_idx, len(self.arr))
        self.test_idx += len(batch)
        return self.__load(batch)

    def reset(self):
        self.train_idx = 0
        self.test_idx = self.train_size
        random.shuffle(self.arr)

    def __next_batch(self, batch_size, first_idx, last_idx):
        remaining = last_idx - first_idx
        if remaining == 0:
            return []
        if batch_size > remaining:
            batch_size = remaining
        result = self.arr[first_idx:first_idx + batch_size]
        return result

    def __load(self, batch):
        if not batch:
            return None, None
        X = np.array([self.x_mapper(item) for item in batch])
        y = np.array([self.y_mapper(item) for item in batch])
        return X, y