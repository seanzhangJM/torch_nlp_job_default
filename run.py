#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 16:49
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : run.py
# @Software: PyCharm
import os
import sys

sys.path.extend(["."])
import re
from d2l import torch as d2l
from torch_nlp_job_default.util.nlp.util import tokenize, Vocab, load_data, read_data


def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine', cache_dir=os.path.join('.', 'data')), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def load_corpus_time_machine(max_tokens=-1):  # @save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


if __name__ == '__main__':
    iters,voca = load_data(2,5)
    import numpy as np
    for X, Y in iters:
        print('X: ', np.array(voca.idx_to_token)[X.data.numpy()], '\nY:', np.array(voca.idx_to_token)[Y.data.numpy()])
    print(voca.token_to_idx)
    print(voca.idx_to_token)
