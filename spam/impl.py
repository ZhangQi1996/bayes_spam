#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'David Zhang'

import logging

import jieba

from spam.base import *
from util import *
from . import CONF, STOP_FILE_PATH, EXPORT_FILES_PATH


class MailFileHelper(FileHelperBase):
    """
    MailFileHelper类，用于辅助处理mail文件内容的读取，内容的单词拆分
    """

    def __init__(self, conf: dict, stop_file_path: str):
        """
        init a MailFileHelper instance
        :param conf: 从conf.ini文件中读取的配置
        :param stop_file_path: 终止文件，用于在从mail文件中读取的内容，划分后的单词若
            其存在于终止文件中，则最终不纳入这个文件的内容单词列表中
        """
        logging.info("initializing an instance of MailFileHelper...")
        self._conf = conf
        self._stop_file_path = stop_file_path
        self._stop_words = None

        # read contents of the stop file
        self._read_stop_file()
        logging.info("has completed the initialization of the instance...")

    def read_file_content(self, fp, encoding=None) -> str:
        """read the contents from the file, and filters the head"""

        if encoding is None:
            encoding = self._conf.get('data_set_encoding')

        with open(fp, 'r', encoding=encoding) as f:
            content = f.read()

        # removes the head of the mail
        return self.mail_file_content_filter(content)

    def _read_stop_file(self, encoding=None):
        """读取停用文件"""
        logging.info("starting to read the stop file %s..." % self._stop_file_path)
        self._stop_words = []

        if encoding is None:
            encoding = self._conf.get('data_set_encoding')

        # read every line in the stop file
        for line in open(self._stop_file_path, 'r', encoding=encoding):
            _ = line.strip()  # trim the line
            if _ == '':
                continue
            self._stop_words.append(_)

        # 将stop_words进行排序, 用于在后面使用二分查找
        # sort the stop_words to use a bi-search subsequently.
        self._stop_words = sort_str_list(self._stop_words)
        logging.info("has completed the read of the stop file...")

    def split_words_from_str(self, s: str) -> list:
        """从s中拆分得到所有不存在于stop_words中的单词(不包括空白字符串)"""
        # override
        ret = []
        # use the jieba lib to split chinese sentences.
        for word in jieba.cut(s):
            word = word.strip()     # trim the word
            # the word not empty and not in stop_words
            if word != '' and find_elem_from_sorted_str_list(word, self._stop_words) is None:
                ret.append(word)

        return ret

    def read_total_word_times(self, file_list: list, log_step_interval: int = 1, encoding=None) -> dict:
        """
        读取这批文件中总共的每个单词的数量

        file_list就是需要读取文件的路径

        step_interval就是读取文件的百分比跨度，其数值介于(0, 100),
        e.g. step_interval = 5, 表示每次的日志跨度为5%
        """
        logging.info("starting to read...")

        ret = {}

        # the step that helps to log the progress.
        step = (len(file_list) // 100) * log_step_interval
        if step == 0:
            step = 1

        if encoding is None:
            encoding = self._conf.get('data_set_encoding')

        for i, fp in enumerate(file_list):
            content = self.read_file_content(fp, encoding)
            for word in self.split_words_from_str(content):     # not includes space str
                ret.setdefault(word, 0)
                ret[word] += 1
            if i % step == 0:
                logging.info("reading progress: %d%%", i * 100 // len(file_list))
        logging.info("reading progress: 100%")

        return ret

    @staticmethod
    def get_all_files_under_dir(dir_path: str) -> list:
        dir_path = dir_path.strip().rstrip('/').rstrip('\\')
        if os.path.isdir(dir_path) is False:
            raise ValueError('the dir_path(%s) you input is not a dir...' % dir_path)

        ret = []

        for f in os.listdir(dir_path):
            ret.append('%s/%s' % (dir_path, f))

        return ret

    @staticmethod
    def mail_file_content_filter(s: str) -> str:
        """
        由于邮件文件内容前一部分是由非中文组成的头部，故这个过滤器就是简单的去除内容的
        前一部分非中文的内容，以及去除剩余空白字符

        s: the content of a mail file
        """

        # 分为两组，第一组就是全部非中文字符，第二组就是剩余部分
        p = re.compile(r'^([^\u4e00-\u9fa5]*)(.*)$', re.MULTILINE | re.DOTALL)
        return p.match(s).group(2).strip()


# an instance, singleton
default_file_helper = MailFileHelper(CONF, STOP_FILE_PATH)


class BayesSpamTrain(BayesSpamTrainBase):
    """
    仅仅用来获取相关参数
    """

    def __init__(self, d_type=np.float32,
                 mail_file_helper=default_file_helper):
        """
        initialize
        :param d_type: 指定内部数据的np.ndarray的数据类型
        :param mail_file_helper: 见MailFileHelper介绍
        """

        # base vars
        self.d_type = d_type
        self._mail_file_helper = mail_file_helper

        # train set
        # in class 0, a map from the word to its occurring times.
        self._train_set_class0_word_times_map = None
        self._train_set_class1_word_times_map = None

        # read conf
        self._read_conf()

    def read_train_set_data(self, file_list: list, clazz: int, *args, **kwargs):
        """you can provide an arg named log_step_interval, see _read_set_data"""
        logging.info("starting to read the data in train set to class %s..." % clazz)
        log_step_interval = kwargs.get('log_step_interval')
        if log_step_interval is None:
            log_step_interval = 1

        d = self._mail_file_helper.read_total_word_times(file_list, log_step_interval)

        if clazz == 0:
            self._train_set_class0_word_times_map = d
        else:
            self._train_set_class1_word_times_map = d

        logging.info("has read the data in train set to class %s..." % clazz)

    def _read_conf(self):
        """
        self._conf = {
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
        """
        self._conf = CONF

    def _train_before(self):
        """计算每个类别单词的总个数，每个类别中的每种单词的概率"""

        _ = check_all_non_none([
            self._train_set_class0_word_times_map,
            self._train_set_class1_word_times_map
        ])
        if _ is not True:
            raise ValueError('the item in index %s is None, '
                             'maybe you do not call read_train_set_xxx method.' % _)

    def train(self, *args, **kwargs):
        """
        exec this method after having read the data of train set, and return a predict model

        可以提供参数p_class_0与p_class_1
        """
        self._train_before()
        p_class_0 = kwargs.get('p_class_0')
        p_class_1 = kwargs.get('p_class_1')
        if p_class_0 is None:
            p_class_0 = 0.5
        if p_class_1 is None:
            p_class_1 = 0.5

        return BayesSpamModel(p_class_0, p_class_1,
                              self._train_set_class0_word_times_map,
                              self._train_set_class1_word_times_map,
                              self.d_type)


class BayesSpamModel(BayesSpamModelBase):
    """
    用于预测
    """

    def __init__(self, p_class_0, p_class_1,
                 train_set_class0_word_times_map,
                 train_set_class1_word_times_map,
                 d_type=np.float32,
                 mail_file_helper=default_file_helper):

        self.p_class_0 = p_class_0
        self.p_class_1 = p_class_1

        self.d_type = d_type
        self._mail_file_helper = mail_file_helper

        self._train_set_class0_word_times = np.array(list(train_set_class0_word_times_map.values()),
                                                     dtype=np.int32)
        self._train_set_class0_word_index_map = {
            key: i for i, key in enumerate(train_set_class0_word_times_map.keys())
        }

        self._train_set_class1_word_times = np.array(list(train_set_class1_word_times_map.values()),
                                                     dtype=np.int32)
        self._train_set_class1_word_index_map = {
            key: i for i, key in enumerate(train_set_class1_word_times_map.keys())
        }

        self._train_set_class0_word_nums = sum(self._train_set_class0_word_times)

        self._train_set_class1_word_nums = sum(self._train_set_class1_word_times)

        self._threshold = -1

        self._read_conf()

    def _read_conf(self):
        """
        self._conf = {
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
        :return: None
        """
        self._conf = CONF

    def read_file_content(self, fp: str) -> str:
        """read the filtered content from the file"""
        return self._mail_file_helper.read_file_content(fp)

    def _get_word_prob_in_train_set_class0(self, word: str, total_words_in_a_file: int) -> float:
        """evaluating formula see https://gitee.com/ChiZhung/study_note/blob/master/ML/naive_bayes.md"""

        # 获取这个单词在类别0中的索引
        i = self._train_set_class0_word_index_map.get(word)

        word_times_in_class0 = 0 if i is None else self._train_set_class0_word_times[i]

        p = (1 + word_times_in_class0) / (total_words_in_a_file + self._train_set_class0_word_nums)
        return p

    def _get_word_prob_in_train_set_class1(self, word: str, total_words_in_a_file: int) -> float:
        """evaluating formula see https://gitee.com/ChiZhung/study_note/blob/master/ML/naive_bayes.md"""

        # 获取这个单词在类别1中的索引
        i = self._train_set_class1_word_index_map.get(word)

        word_times_in_class1 = 0 if i is None else self._train_set_class1_word_times[i]

        p = (1 + word_times_in_class1) / (total_words_in_a_file + self._train_set_class1_word_nums)
        return p

    def _get_word_prob_in_train_set(self, word: str, total_words_in_a_file: int, clazz: int):
        if clazz == 0:
            return self._get_word_prob_in_train_set_class0(word, total_words_in_a_file)
        return self._get_word_prob_in_train_set_class1(word, total_words_in_a_file)

    def _cmp_prob_of_one(self, data: list) -> tuple:
        """
        计算一个mail（由其单词数组代表）分别属于normal, spam类别的概率'

        evaluates respectively normal and spam prob of the mail represented by it's words' array.
        """

        # 由于一个文件中存在重复单词，故使用缓存
        word_cache_dict_class0 = {}
        word_cache_dict_class1 = {}

        total_words_in_a_file = len(data)

        # evaluate p(w=wij|c=0 or 1) per word.
        for word in data:
            word_cache_dict_class0.setdefault(
                word,
                self._get_word_prob_in_train_set_class0(word, total_words_in_a_file)
            )
            word_cache_dict_class1.setdefault(
                word,
                self._get_word_prob_in_train_set_class1(word, total_words_in_a_file)
            )

        # ndarray
        file_p_class0 = np.zeros(total_words_in_a_file, dtype=self.d_type)
        file_p_class1 = np.zeros(total_words_in_a_file, dtype=self.d_type)

        for i in range(total_words_in_a_file):
            file_p_class0[i] = word_cache_dict_class0.get(data[i])
            file_p_class1[i] = word_cache_dict_class1.get(data[i])

        # top threshold
        if file_p_class0.size < self._threshold:
            file_p_class0 = top_k(file_p_class0, -1)
            file_p_class1 = top_k(file_p_class1, -1)
        else:
            file_p_class0 = top_k(file_p_class0, self._threshold)
            file_p_class1 = top_k(file_p_class1, self._threshold)

        # formula: Sigma(1,n){log(p(w=wij|c=x)} + log(p(c=x))
        # int val
        file_p_class0 = np.sum(np.log(file_p_class0)) + np.log(self.p_class_0)
        file_p_class1 = np.sum(np.log(file_p_class1)) + np.log(self.p_class_1)

        return file_p_class0, file_p_class1

    def set_threshold(self, threshold=-1):
        """设置在预测时采用文本的前高频率threshold个词，-1表示所有词"""
        self._threshold = threshold

    def predict_one(self, data: list) -> int:
        """data就是一个文件的内容单词列表, 返回预测结果"""
        return self.predict_tup_one(
            self._cmp_prob_of_one(data)
        )

    def predict_one_file(self, fp, encoding=None) -> int:
        """if not specifies encoding, it will be set according to conf.ini--data_set_encoding"""
        words = self._mail_file_helper.read_file_words(fp, encoding)
        return self.predict_one(words)

    def predict_one_bytes(self, data: bytes, encoding=None) -> int:
        """if not specifies encoding, it will be set according to conf.ini--data_set_encoding"""
        if encoding is None:
            encoding = self._conf.get('data_set_encoding')

        content = bytes2str(data, encoding)
        filtered_content = self._mail_file_helper.mail_file_content_filter(content)
        words = self._mail_file_helper.split_words_from_str(filtered_content)
        return self.predict_one(words)

    def predict_many(self, datas: list) -> list:
        """datas就是多个文件的内容2dim单词列表, 返回预测结果"""
        return self.predict_tup_many(
            self._cmp_prob_of_many(datas)
        )

    def predict_many_files(self, file_list: list, encoding=None) -> list:
        """if not specifies encoding, it will be set according to conf.ini--data_set_encoding"""
        return [self.predict_one_file(fp, encoding) for fp in file_list]

    def predict_many_bytes(self, datas: list, encoding=None) -> list:
        """if not specifies encoding, it will be set according to conf.ini--data_set_encoding"""
        return [self.predict_one_bytes(data, encoding) for data in datas]

    @staticmethod
    def acc(y_hat: list, y: list):
        """计算正确率"""
        y_hat = np.array(y_hat).flatten()
        y = np.array(y).flatten()

        if y_hat.size != y.size:
            raise TypeError('the size of y_hat(%s) is not equal to y(%s)' % (y_hat.size, y.size))

        return np.sum(np.equal(y_hat, y)) / y_hat.size

    @staticmethod
    def _check_fps(fps):
        split = re.split(r'\s+', fps)
        if len(split) != 2:
            raise ValueError('fps must consist of two paths, but it is %s' % fps)
        return split[0], split[1]

    def export_model(self, fps=EXPORT_FILES_PATH, encoding='utf-8'):
        """
        将 train_set_class0_word_times_map与
         train_set_class1_word_times_map分别放置在文件model_class0.txt model_class1.txt中
        """
        file_class0, file_class1 = self._check_fps(fps)

        logging.info("exporting the model...")

        with open(file_class0, 'w', encoding=encoding) as f:
            for word, i in self._train_set_class0_word_index_map.items():
                f.write("%s %s\n" % (word, self._train_set_class0_word_times[i]))

        with open(file_class1, 'w', encoding=encoding) as f:
            for word, i in self._train_set_class1_word_index_map.items():
                f.write("%s %s\n" % (word, self._train_set_class1_word_times[i]))

        logging.info("has completed the export...")

    @staticmethod
    def import_model(fps=EXPORT_FILES_PATH,
                     p_class_0=0.5,
                     p_class_1=0.5,
                     d_type=np.float32,
                     mail_file_helper=default_file_helper,
                     encoding='utf-8'):
        """import the spam predict model from the specified files"""
        file_class0, file_class1 = BayesSpamModel._check_fps(fps)

        logging.info("importing the model...")

        train_set_class0_word_times_map = {}
        train_set_class1_word_times_map = {}

        def load(fp, d):
            for i, line in enumerate(open(fp, 'r', encoding=encoding)):
                try:
                    line = line.strip()
                    word, times = re.split(r'\s+', line)
                    times = int(times)

                    d[word] = times
                except:
                    raise Exception('parse the file %s in failure, line %s: \'%s\''
                                    % (fp, i + 1, line))

        load(file_class0, train_set_class0_word_times_map)
        load(file_class1, train_set_class1_word_times_map)

        logging.info("has completed the import...")

        return BayesSpamModel(p_class_0,
                              p_class_1,
                              train_set_class0_word_times_map,
                              train_set_class1_word_times_map,
                              d_type,
                              mail_file_helper)

    def evaluate(self, file_list: list, clazz: int = None, y: list = None, encoding=None) -> float:
        """
        提供一个文件列表以及其对应的真实类别，0是normal，1是spam

        e.g.
        evaluate(['1.txt', '2.txt', '3.txt'], clazz=0, encoding='utf-8')
        that means all the true class of 1-3.txt are class 0(normal)

        evaluate(['1.txt', '2.txt', '3.txt'], y=[0, 1, 0], encoding='utf-8')
        that means the true class of 1-3.txt is class 0, class 1, class 0 respectively.
        """
        if isinstance(clazz, int):
            y = np.zeros(len(file_list)) + clazz
        if y is None:
            raise ValueError("you must provide either an arg clazz or y...")
        y_hat = self.predict_many_files(file_list, encoding)
        return self.acc(y_hat, y)
