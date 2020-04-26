import jieba
from util import read_conf, mail_file_content_filter
import logging

CONF_FILE_PATH = '../conf.ini'
STOP_FILE_PATH = '../data/stop_words.txt'
ENCODING = 'utf-8'

CONF = read_conf(CONF_FILE_PATH, ENCODING)

logging.basicConfig(
    stream=CONF.get('stream'),
    level=CONF.get('log_level'),
    datefmt=CONF.get('log_date_fmt'),
    format=CONF.get('log_format')
)
logging.info("读取配置并配置记录日志成功...")
