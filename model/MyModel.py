#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 14:03
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : MyModel.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_nlp_job_default.util.log_config import get_logger

logger = get_logger()


class RNNModel(nn.Module):
    """循环神经网络模型"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # nn.Embedding
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers=num_layers, dropout=dropout)

    def forward(self, X):
        # X shape:[batch_size,time_steps,embed_size]
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        # output:[time_steps,batch_size,num_hiddens]
        # state[0]:[2*num_layers,batch_size,num_hiddens] if bidirectional else [num_layers,batch_size,num_hiddens]
        return output, state


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers=num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)
        # output shape:[time_steps,batch_size,num_hiddens]
        output = self.dense(output).permute(1, 0, 2)
        return output, state
