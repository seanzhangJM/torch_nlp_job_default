#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 14:29
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : util.py
# @Software: PyCharm

import os
import re
import torch
import numpy as np
import random
from pytorch_default_demo.util.log_config import get_logger

logger = get_logger()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


def process_en(text):
    """英文字符串预处理"""
    text = re.sub('[!！]+', " ", text)
    text = re.sub('[?？]+', " ", text)
    text = re.sub("[\"#\\$%&()◎—""－～（）∩*+,-./:;：;；<=>@，。★、…【】《》“”‘’""·[\\]^_`{|}~＃\\\]+", " ", text)
    text = re.sub("[とに一緒にèéêóも]+", " ", text)
    text = re.sub("[0-9]+", " ", text)
    filters = ['\t', '\n', '\x97', '\x96']
    text = re.sub("|".join(filters), ' ', text)
    return text.strip().lower()


def process_zh(text):
    """中文行字符串预处理"""
    text = re.sub('[!！]+', "", text)
    text = re.sub('[?？]+', "", text)
    text = re.sub("[a-zA-Z#$%&\'()◎—""－～（）∩*+,-./:;：;；<=>@，。★、…【】《》“”‘’""·[\\]^_`{|}~]+", "", text)
    text = re.sub("[とに一緒にèéêóも]+", "", text)
    text = re.sub(r"\W+", "", text)  # 去掉非单词
    text = re.sub("[0-9]+", "", text)
    text = re.sub("\s+", "", text)
    return text
