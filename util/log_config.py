#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 14:31
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : log_config.py
# @Software: PyCharm

import os
import sys
import datetime
import logging
import logging.config
from pytorch_default_demo.util.root_config import LOG_DIR, LOG_LEVEL, CONSOLE_LEVEL


# 项目文件
BASE_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../")
# base dir 加入sys.path当中
sys.path.insert(0, BASE_DIR)


# 时区,东八区
def beijing(sec, what):
    """
    :param sec:
    :param what:
    :return: ex:time.struct_time(tm_year=2021, tm_mon=12, tm_mday=27, tm_hour=15, tm_min=24, tm_sec=3, tm_wday=0, tm_yday=361, tm_isdst=-1)
    """
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


logging.Formatter.converter = beijing


def get_logger(log_file_name='info'):
    conf = {'version': 1,
            'disable_existing_loggers': False,
            'incremental': False,
            'formatters': {
                'verbose': {
                    'format': '|%(asctime)s|%(name)s|%(filename)s|%(lineno)d|%(levelname)s|%(processName)s|%(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'},
                'simple': {
                    'format': "|%(levelname)s|%(message)s"
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': CONSOLE_LEVEL,
                    'formatter': 'verbose',
                    'stream': 'ext://sys.stdout'
                },
                'info_file_handler': {
                    'class': 'logging.handlers.TimedRotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'verbose',
                    'filename': os.path.join(LOG_DIR, '%s.log' % log_file_name),
                    'when': 'W6',
                    'interval': 1,
                    'backupCount': 7,
                    'encoding': "utf-8"
                },
                'error_file_handler': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'verbose',
                    'filename': os.path.join(LOG_DIR, 'error.log'),
                    'maxBytes': 1 * 1024 * 1024,
                    'backupCount': 7,
                    'encoding': "utf8"
                }
            },
            'loggers': {
                'my_module': {
                    'handlers': ['console'],
                    'level': LOG_LEVEL,
                    'propagate': 'no'
                }
            },
            'root':
                {
                    'handlers': ['console', 'info_file_handler', 'error_file_handler'],

                    'level': LOG_LEVEL
                }
            }
    logging.config.dictConfig(conf)
    logger = logging.getLogger(__name__)
    return logger


if __name__ == '__main__':
    import pprint
    pprint.pprint(sys.path)
