#!/usr/bin/python

from abc import ABC, abstractmethod
import numpy as np
import re


class BayesSpamTrainBase(ABC):

    d_type = np.float32

    @abstractmethod
    def read_train_set_data(self, file_list: list, clazz: int, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        '''exec this method after having read the data of train set, and return a predict model'''
        pass


class BayesSpamModelBase(ABC):

    # the probability of class 0 (default: normal)
    p_class_0 = 0.5

    # the probability of class 1 (default: spam)
    p_class_1 = 0.5

    d_type = np.float32

    def _cmp_prob_of_many(self, datas: np.ndarray):
        '''
        datas is a np array whose dim is 2, it consists of a series of words representing many files' contents.

        E.g. datas = np.array([
            ['I', love', 'you'],    # representing the 1st file's content
            ['I', 'hate', 'you']    # representing the 2nd file's content
        ])
        '''
        if np.ndim(datas) != 2:
            raise TypeError("the dim of datas expects 2, but %s" % np.ndim(datas))

        n, m = np.shape(datas)
        m = 2

        ret = np.zeros(shape=(n, m), dtype=self.d_type)

        for i in range(n):
            ret[i][0], ret[i][1] = self._cmp_prob_of_one(datas[i])

        return ret

    @abstractmethod
    def _cmp_prob_of_one(self, data: np.ndarray):
        '''
        data is a np array whose dim is 1, it consists of a series of words representing a file's content.

        E.g. data = np.array(['I', love', 'you'])

        return np.array([prob_0, prob_1])
        '''
        pass

    @staticmethod
    def predict_tup_one(tup):
        '''tup just like [prob_0, prob_1], it just is used to judge its class(0 or 1)'''
        return 0 if tup[0] > tup[1] else 1

    @staticmethod
    def predict_tup_many(datas):
        '''
        E.g.
        [(0.6, 0.4), (0.2, 0.8)] -> [0, 1]
        '''
        ret = np.zeros(shape=len(datas), dtype=np.int32)

        for i, tup in enumerate(datas):
            ret[i] = BayesSpamModelBase.predict_tup_one(tup)

        return ret

    @abstractmethod
    def predict_one(self, data: np.ndarray):
        pass

    @abstractmethod
    def predict_many(self, datas: np.ndarray):
        pass

    @abstractmethod
    def read_test_set_data(self, file_list: list, clazz: int, *args, **kwargs):
        pass


class MailFileHelperBase(ABC):
    @abstractmethod
    def read_data(self, fp):
        pass

    @abstractmethod
    def split_words_from_str(self, s: str):
        '''this method will split a string into some words'''
        return re.split(r'\s+', s.strip())
