#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'David Zhang'

from configparser import ConfigParser
from sys import stdout, stderr
import os
import re
import numpy as np

PROJ_NAME = 'bayes_spam'


def get_proj_abs_path():
    p = os.path.abspath(".").replace('\\', '/')
    match = re.compile('(.*%s)' % PROJ_NAME).match(p)
    return match.group(1)


PROJ_BASE_PATH = get_proj_abs_path()


def sort_str_list(l: list):
    return sorted(l)


def find_elem_from_sorted_str_list(e: str, l: list):
    '''
    查找成功返回该元素索引，否则返回None

    finds the target element and return its index, if any, otherwise None.
    '''
    # 采用二分查找(uses bi-find)
    lo, hi = 0, len(l) - 1
    while lo < hi:
        m = (lo + hi) >> 1
        if l[m] == e:
            return m
        elif l[m] > e:
            hi = m - 1
        else:
            lo = m + 1
    return None


def bytes2str(data: bytes, encoding='utf-8'):
    return data.decode(encoding)


def check_all_non_none(it):
    '''return index, if any one is None, else True.'''
    for i, item in enumerate(it):
        if item is None:
            return i
    return True


def read_conf(fp='conf.ini', encoding='utf-8') -> dict:
    '''read confs from fp, and return {encoding, train_set_spam_path, ...}'''
    conf = ConfigParser()
    conf.read(fp, encoding=encoding)

    conf_dict = {}
    ret = {}

    for sec in conf.sections():
        conf_dict.setdefault(sec, conf.options(sec))

    for sec in conf_dict.keys():
        for op in conf_dict.get(sec):
            ret[op] = conf.get(sec, op, raw=True)

    if ret.get('stream_type') == '1':
        ret.setdefault('stream', stdout)
    elif ret.get('stream_type') == '2':
        ret.setdefault('stream', stderr)
    else:
        log_file = "%s/%s" % (PROJ_BASE_PATH, ret.get('log_file'))
        encoding = ret.get('log_file_encoding')
        ret.setdefault('stream', open(log_file, mode='w', encoding=encoding))

    return ret


def top_k(l, k: int, result_order=False):
    """
    返回list中最大的k的元素的值，若找到返回list
    result_order表示是否结果有序，False表示不保证有序
    注意：这个方法可能会排序所提供的数组
    """
    if k > len(l) or k < -1:
        raise TypeError("the arg l or k is illegal, len of l is %s, k is %s" % (len(l), k))

    if not isinstance(l, np.ndarray):
        l = np.array(l)
    l = l.flatten()

    if k == -1 and result_order is False:
        return l

    l.sort()
    return l[-k:]