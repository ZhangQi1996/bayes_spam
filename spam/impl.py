#!/usr/bin/python

import numpy as np
import logging
from util import read_conf, sort_str_list, \
    find_elem_from_sorted_str_list, mail_file_content_filter, check_all_non_none
from spam.base import *
import jieba
from . import CONF, STOP_FILE_PATH


class MailFileHelper(MailFileHelperBase):

    def __init__(self, conf: dict, stop_file_path: str):
        self.conf = conf
        self.stop_file_path = stop_file_path
        self.stop_words = None

        self._read_stop_file()

    def read_data(self, fp):
        '''read the contents from the file'''
        with open(fp, 'r', encoding=self.conf.get('data_set_encoding')) as f:
            content = f.read()

        # removes the head of the mail
        return mail_file_content_filter(content)

    def _read_stop_file(self):
        '''读取停用文件'''
        logging.info("开始读取终止单词文件%s..." % self.stop_file_path)
        self.stop_words = []

        for line in open(self.stop_file_path, 'r', encoding=self.conf.get('data_set_encoding')):
            _ = line.strip()
            if _ == '':
                break
            self.stop_words.append(_)

        # 将stop_words进行排序, 用于在后面使用二分查找
        self.stop_words = sort_str_list(self.stop_words)
        logging.info("读取终止单词文件完成...")

    def split_words_from_str(self, s: str):
        '''从s中拆分得到所有不存在于stop_words中的单词'''
        ret = []
        for word in jieba.cut(s):
            if find_elem_from_sorted_str_list(word, self.stop_words) is not None:
                ret.append(word)

        return ret

    def read_set_data(self, file_list: list, log_step_interval: int = 1):
        """
        读取这个类别中的每个单词的数量

        file_list就是需要读取文件的路径

        step_interval就是读取文件的百分比跨度，其数值介于(0, 100),
        e.g. step_interval = 5, 表示每次的日志跨度为5%
        """
        ret = {}

        step = (len(file_list) // 100) * log_step_interval
        if step == 0:
            step = 1

        for i, fp in enumerate(file_list):
            content = self.read_data(fp)
            for word in self.split_words_from_str(content):
                ret.setdefault(word, 0)
                ret[word] += 1
            if i % step == 0:
                logging.info("读取进度: %d%%", i * 100 // len(file_list))
        logging.info("读取进度: 100%")

        return ret


default_file_helper = MailFileHelper(CONF, STOP_FILE_PATH)


class BayesSpamTrain(BayesSpamTrainBase):

    def __init__(self, d_type=np.float32,
                 mail_file_helper=default_file_helper):

        # base vars
        self.d_type = d_type
        self.mail_file_helper = mail_file_helper

        # train set
        self.train_set_class0_word_times_map = None
        self.train_set_class1_word_times_map = None

        # read conf
        self._read_conf()

    def read_train_set_data(self, file_list: list, clazz: int, *args, **kwargs):
        '''you can provide an arg named log_step_interval, see _read_set_data'''
        logging.info("开始读取类别%s的训练数据集文件数据..." % clazz)
        log_step_interval = kwargs.get('log_step_interval')
        if log_step_interval is None:
            log_step_interval = 1

        d = self.mail_file_helper.read_set_data(file_list, log_step_interval)

        if clazz == 0:
            self.train_set_class0_word_times_map = d
        else:
            self.train_set_class1_word_times_map = d

        logging.info("读取类别%s的训练数据集文件数据完成..." % clazz)

    def _read_conf(self):
        '''
        self.conf = {
            encoding
            train_set_normal_path
            train_set_spam_path
            train_set_encoding
            stream_type
            log_file
            log_level
            log_format
            log_date_fmt
            stream
        }
        :return:
        '''
        self.conf = CONF

    def _train_before(self):
        '''计算每个类别单词的总个数，每个类别中的每种单词的概率'''

        _ = check_all_non_none([
            self.train_set_class0_word_times_map,
            self.train_set_class1_word_times_map
        ])
        if _ is not True:
            raise ValueError('the item in index %s is None' % _)

    def train(self, *args, **kwargs):
        self._train_before()
        p_class_0 = kwargs.get('p_class_0')
        p_class_1 = kwargs.get('p_class_1')
        if p_class_0 is None:
            p_class_0 = 0.5
        if p_class_1 is None:
            p_class_1 = 0.5

        return BayesSpamModel(p_class_0, p_class_1,
                              self.train_set_class0_word_times_map,
                              self.train_set_class1_word_times_map,
                              self.d_type)


class BayesSpamModel(BayesSpamModelBase):

    def __init__(self, p_class_0, p_class_1,
                 train_set_class0_word_times_map,
                 train_set_class1_word_times_map,
                 d_type=np.float32,
                 mail_file_helper=default_file_helper):

        self.p_class_0 = p_class_0
        self.p_class_1 = p_class_1

        self.d_type = d_type
        self.mail_file_helper = mail_file_helper

        self.train_set_class0_word_times = np.array(list(train_set_class0_word_times_map.values()),
                                                    dtype=self.d_type)
        self.train_set_class0_word_index_map = {
            key: i for i, key in enumerate(train_set_class0_word_times_map.keys())
        }

        self.train_set_class1_word_times = np.array(list(train_set_class1_word_times_map.values()),
                                                    dtype=self.d_type)
        self.train_set_class1_word_index_map = {
            key: i for i, key in enumerate(train_set_class1_word_times_map.keys())
        }

        self.train_set_class0_word_nums = sum(self.train_set_class0_word_times)
        self.train_set_class0_word_prob = self.train_set_class0_word_times / self.train_set_class0_word_nums

        self.train_set_class1_word_nums = sum(self.train_set_class1_word_times)
        self.train_set_class1_word_prob = self.train_set_class1_word_times / self.train_set_class1_word_nums

    def _read_conf(self):
        '''
        self.conf = {
            encoding
            train_set_normal_path
            train_set_spam_path
            train_set_encoding
            stream_type
            log_file
            log_level
            log_format
            log_date_fmt
            stream
        }
        :return:
        '''
        self.conf = CONF

    def read_test_set_data(self, file_list: list, clazz: int, *args, **kwargs):
        '''you can provide an arg named log_step_interval, see _read_set_data'''
        logging.info("开始读取类别%s的测试数据集文件数据..." % clazz)
        log_step_interval = kwargs.get('log_step_interval')
        if log_step_interval is None:
            log_step_interval = 1

        d = self.mail_file_helper.read_set_data(file_list, log_step_interval)

        if clazz == 0:
            # TODO
            pass
        else:
            # TODO
            pass
        logging.info("读取类别%s的测试数据集文件数据完成..." % clazz)

    def _get_word_prob_in_train_set_class0(self, word: str):
        i = self.train_set_class0_word_index_map.get(word)
        if i is None:
            return 0
        return self.train_set_class0_word_prob[i]

    def _get_word_prob_in_train_set_class1(self, word: str):
        i = self.train_set_class1_word_index_map.get(word)
        if i is None:
            return 0
        return self.train_set_class1_word_prob[i]

    def _get_word_prob_in_train_set(self, word: str, clazz: int):
        if clazz == 0:
            return self._get_word_prob_in_train_set_class0(word)
        return self._get_word_prob_in_train_set_class1(word)

    def _cmp_prob_of_one(self, data: np.ndarray):
        pass

    def predict_one(self, data: np.ndarray):
        pass

    def predict_many(self, datas: np.ndarray):
        pass
