#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/1/4 15:43
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : util.py
# @Software: PyCharm
import collections
import random
import torch
import os
from torch_nlp_job_default.util.root_config import TRAIN_DATA_DIR
from torch_nlp_job_default.util.util import process_zh
from pytorch_default_demo.util.log_config import get_logger

logger = get_logger()


def read_data():
    """将时间机器数据集加载到文本行的列表中"""
    # 这个地址可以配到config里面
    # TODO 这把读取的是一个文件，实时上，我们经常处理的是一个文件夹下的一批文件
    data_path = os.path.join(TRAIN_DATA_DIR, 'zjm.csv')
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 这边是英文的正则处理方式，可以百度到中文的正则处理方式！[re.sub('[^A-Za-z]+', ' ', line).strip() for line in lines]
    # 根据具体需求，可以去除一些特殊字符，英文字符，标点符号，数字，繁体字
    # 更进一步用分词工具，去除停用词
    # TODO 这边用yield生成数据是不是更好一点，[]一下子把所有数据都抛出了？随后的生成batch_size的数据都用generator的方式这样可以最大节省内存。
    return [process_zh(line) for line in lines]


def tokenize(lines, token='word', language="en"):
    """
    将文本行拆分为单词或字符词元，可以引入结巴分词
    :param lines:
    :param token:
    :param language: 中英文标识
    :return:
    """
    if language == "en":
        if token == 'word':
            # TODO 同样这边也可以用yield 生成数据以节约内存
            return [line.split() for line in lines]
        elif token == 'char':
            # TODO 同样这边也可以用yield 生成数据以节约内存
            return [list(line) for line in lines]
        else:
            logger.error('错误：未知词元类型：' + token)
    else:
        # TODO 中文的处理，包括分词和不分词，深度学习中经常不分词，因为很多场景中，除非竞赛时候，不然分词并没有对结果改善很多
        if token == 'word':
            # TODO 这边使用分词工具分词后yield 生成数据以节约内存
            pass
            # return [line.split() for line in lines]
        elif token == 'char':
            # TODO 同样这边也可以用yield 生成数据以节约内存
            return [list(line) for line in lines]
        else:
            logger.error('错误：未知词元类型：' + token)


class Vocab:
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=['<pad>', '<bos>', '<eos>']):
        r"""
        :param tokens: 词元列表,
        ex:
            [['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him'],
            ['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and'],
            ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']]
        :param min_freq:最小出现词频
        :param reserved_tokens:保留字，pad填充，bos开始，eos结尾
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            # 因为这边已经把token_freqs 进行统计倒排，所以遇到频率小于min_freq直接break
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # 使用递归方式进行getitem，这样既可以访问一列数据的idx，也可以访问一个单词的idx
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """
    统计词元的频率
    :param tokens: 词元分词后的列表
    :return: 统计字典dict
    """
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus(max_tokens=-1):
    """
    注意：这里处理的是文章
    :param max_tokens: 截取的最大单词数，默认-1是获取全部文章单词
    :return: 词元的数字表达列表，词元词典
    """
    lines = read_data()
    # TODO 可以考虑加入停用词过滤，使得训练数据符合模型输入分布要求
    tokens = tokenize(lines, 'char')
    #预训练的时候，也可以直接用人家的词典，人家的词向量,这样会节省很多基础工作，用大组织已经公开的文本信息~
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中，因为这里处理的是一个完整的作品，每一行是连贯的
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器,这里可以改写的，这边数据的处理的都是长序列，譬如说论文，小说这些。如果是评论的话就没这么复杂了"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        # 这边加载的是网络数据集，其实我们可以同样的加载本地的一个作品文章，譬如保存在本地文件里的.csv文本,这里我做了一个zjm.csv为列
        self.corpus, self.vocab = load_corpus(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data(batch_size, num_steps, use_random_iter=False, max_tokens=-1):
    """对外的数据接口，返回长文本数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


# **************** encoder-decoder 架构的数据处理功能函数 ****************
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len
