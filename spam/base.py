#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'David Zhang'

import re
from abc import ABC, abstractmethod

import numpy as np


class BayesSpamTrainBase(ABC):

    d_type = np.float32

    @abstractmethod
    def read_train_set_data(self, file_list: list, clazz: int, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """exec this method after having read the data of train set, and return a predict model"""
        pass


class BayesSpamModelBase(ABC):

    # the probability of class 0 (default: normal)
    p_class_0 = 0.5

    # the probability of class 1 (default: spam)
    p_class_1 = 0.5

    d_type = np.float32

    def _cmp_prob_of_many(self, datas: list):
        """
        datas is a list whose dim is 2, it consists of a series of words representing many files' contents.

        E.g. datas = [
            ['I', love', 'you'],    # representing the 1st file's content
            ['I', 'hate', 'you']    # representing the 2nd file's content
        ]
        """
        n, m = len(datas), 2

        ret = np.zeros(shape=(n, m), dtype=self.d_type)

        for i in range(n):
            ret[i][0], ret[i][1] = self._cmp_prob_of_one(datas[i])

        return ret

    @abstractmethod
    def _cmp_prob_of_one(self, data: list):
        """
        data is a np array whose dim is 1, it consists of a series of words representing a file's content.

        E.g. data = ['I', love', 'you']

        return np.array([prob_0, prob_1])
        """
        pass

    @staticmethod
    def predict_tup_one(tup):
        """tup just like [prob_0, prob_1], it just is used to judge its class(0 or 1)"""
        if np.ndim(tup) != 1:
            raise TypeError('the dim of datas must be 1, but %s' % np.ndim(tup))

        if len(tup) != 2:
            raise TypeError("the len of tup must be 2, but %s" % len(tup))

        return 0 if tup[0] > tup[1] else 1

    @staticmethod
    def predict_tup_many(datas):
        """
        E.g.
        [(0.6, 0.4), (0.2, 0.8)] -> [0, 1]
        """
        if np.ndim(datas) != 2:
            raise TypeError('the dim of datas must be 2, but %s' % np.ndim(datas))

        return np.argmax(datas, axis=0)

    @abstractmethod
    def predict_one(self, data: list):
        pass

    @abstractmethod
    def predict_many(self, datas: list):
        pass


class FileHelperBase(ABC):
    @abstractmethod
    def read_file_content(self, fp: str, encoding: str):
        pass

    def split_words_from_str(self, s: str):
        """this method will split a string into some words"""
        return re.split(r'\s+', s.strip())

    def read_file_words(self, fp: str, encoding=None) -> list:
        """read total words in the file"""
        content = self.read_file_content(fp, encoding)
        return self.split_words_from_str(content)
