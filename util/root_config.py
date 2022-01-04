#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 14:31
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : root_config.py
# @Software: PyCharm

import os
import sys
import pathlib

# ************************ 全局路径信息 ************************
BASE_DIR = pathlib.Path(os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../"))
NAS_DIR = BASE_DIR / 'nas'
DATA_DIR = NAS_DIR / 'data'
MODEL_DIR = NAS_DIR / 'model_save'
PICTURE_DIR = NAS_DIR / 'picture'
LOG_DIR = NAS_DIR / 'log'

TRAIN_DATA_DIR = BASE_DIR / 'data'
# ************************ 日志级别 ************************
LOG_LEVEL = 'DEBUG'
CONSOLE_LEVEL = 'INFO'
# ************************ 全局日期格式 ************************
PYTHON_TIME_FOMMAT = '%Y-%m-%d T %H:%M:%S'

if __name__ == '__main__':
    print(BASE_DIR)
    import datetime
    t = datetime.datetime.now()
    print(datetime.datetime.strftime(t,PYTHON_TIME_FOMMAT))