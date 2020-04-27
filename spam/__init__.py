#!/usr/bin/python

from util import read_conf
import logging
from util import PROJ_BASE_PATH

CONF_FILE_PATH = '%s/conf/conf.ini' % PROJ_BASE_PATH
EXPORT_FILES_PATH = '%s/conf/model_class0.txt %s/conf/model_class1.txt' % (PROJ_BASE_PATH, PROJ_BASE_PATH)
STOP_FILE_PATH = '%s/data/stop_words.txt' % PROJ_BASE_PATH
# specify the encoding of non-data files
ENCODING = 'utf-8'

CONF = read_conf(CONF_FILE_PATH, ENCODING)

logging.basicConfig(
    stream=CONF.get('stream'),
    level=CONF.get('log_level'),
    datefmt=CONF.get('log_date_fmt'),
    format=CONF.get('log_format')
)
logging.info("has read the conf and set the logging successfully...")
