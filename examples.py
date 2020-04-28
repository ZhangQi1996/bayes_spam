#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'David Zhang'

from spam.impl import *
import logging


def train() -> BayesSpamModel:
    # 获取训练数据集中属于正常邮件的文件路径
    # get files belonging to normal email in train set
    train_normal = MailFileHelper.get_all_files_under_dir('data/train/normal')
    # 获取训练数据集中属于垃圾邮件的文件路径
    # get files belonging to spam email in train set
    train_spam = MailFileHelper.get_all_files_under_dir('data/train/spam')

    # get a train instance
    bayes_spam = BayesSpamTrain()
    # read data from train set
    # the arg log_step_interval is used to help logging
    bayes_spam.read_train_set_data(train_normal, 0, log_step_interval=5)
    bayes_spam.read_train_set_data(train_spam, 1, log_step_interval=5)

    model = bayes_spam.train()

    # export the model into specified/default local files.
    model.export_model()
    return model


def test(model=None):

    if model is None:
        # import the model from specified/default local files.
        model = BayesSpamModel.import_model()

    test_normal = MailFileHelper.get_all_files_under_dir('data/test/normal')
    test_spam = MailFileHelper.get_all_files_under_dir('data/test/spam')

    file_list = np.concatenate((np.array(test_normal), np.array(test_spam))).flatten()
    y = np.concatenate((np.zeros(len(test_normal)), np.ones(len(test_spam)))).flatten()

    acc = model.evaluate(file_list, y=y)
    return acc


if __name__ == '__main__':
    model = BayesSpamModel.import_model()
    # 设置预测时使用到的高概率词数量
    model.set_threshold(-1)
    logging.info(test(model))

    # train()