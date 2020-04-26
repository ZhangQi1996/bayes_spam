#!/usr/bin/python

from configparser import ConfigParser
from sys import stdout, stderr
import re
import os


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


def read_conf(fp='conf.ini', encoding='utf-8'):
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
        log_file = ret.get('log_file')
        encoding = ret.get('log_file_encoding')
        ret.setdefault('stream', open(log_file, mode='w', encoding=encoding))

    return ret


def bytes2str(data: bytes, encoding='utf-8'):
    return data.decode(encoding)


def mail_file_content_filter(s: str):
    '''
    由于邮件文件内容前一部分是由非中文组成的头部，故这个过滤器就是简单的去除内容的
    前一部分非中文的内容，以及去除剩余空白字符

    s: the content of a mail file
    '''

    # 分为两组，第一组就是全部非中文字符，第二组就是剩余部分
    p = re.compile(r'^([^\u4e00-\u9fa5]*)(.*)$', re.MULTILINE | re.DOTALL)
    return p.match(s).group(2).strip()


def get_all_files_under_dir(dir_path: str):
    dir_path = dir_path.strip().rstrip('/').rstrip('\\')
    if os.path.isdir(dir_path) is False:
        raise ValueError('the dir_path(%s) you input is not a dir...' % dir_path)

    ret = []

    for f in os.listdir(dir_path):
        ret.append('%s/%s' % (dir_path, f))

    return ret


def check_all_non_none(it):
    '''return index, if any one is None, else True.'''
    for i, item in enumerate(it):
        if item is None:
            return i
    return True
