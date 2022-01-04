#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 14:29
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : util.py
# @Software: PyCharm

import os
from pytorch_default_demo.util.log_config import get_logger

logger = get_logger()


def mkdir(path_name):
    """
    创建目录
    :param path_name:
    :return:
    """
    if not isinstance(path_name, str):
        path_name = str(path_name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    else:
        logger.info("{}目录已经存在".format(path_name))
