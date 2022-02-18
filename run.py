#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 16:49
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : run.py
# @Software: PyCharm

import os
import sys
import d2l.torch as d2l
import torchvision
from torch.utils import data
from torch_nlp_job_default.model.MyModel import RNNModel
from torch_nlp_job_default.util.nlp.rnn.tt import train
from torch_nlp_job_default.util.general_process import try_gpu
from torch_nlp_job_default.util.nlp.util import tokenize, Vocab, load_data, read_data
import torch.nn as nn
import torch
from torchvision import transforms

sys.path.extend(["."])


def train_rnnmodel_main():
    global device
    batch_size, num_steps = 2, 5
    train_iter, vocab = load_data(batch_size, num_steps)
    vocab_size, num_hiddens, device = len(vocab), 256, try_gpu()
    num_epochs, lr = 500, 1
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = RNNModel(gru_layer, len(vocab))
    # devices = [try_gpu(i) for i in range(2)]
    # model = nn.DataParallel(model,devices).cuda()
    model = model.to(device)
    train(model, train_iter, vocab, lr, num_epochs, device)
d2l.train_ch8()

def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""

    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net


def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # mnist_train = torchvision.datasets.FashionMNIST(
    #     root="./torch_nlp_job_default/data", train=True, transform=trans, download=True)
    # mnist_test = torchvision.datasets.FashionMNIST(
    #     root="./torch_nlp_job_default/data", train=False, transform=trans, download=True)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=d2l.get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=d2l.get_dataloader_workers()))

def train_mul_gpu(net, num_gpus, batch_size, lr):
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = [try_gpu(i) for i in range(num_gpus)]

    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    # 在多个GPU上设置模型
    net = nn.DataParallel(net, device_ids=devices).cuda()
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')


def train_cnnmodel_main():
    net = resnet18(10)
    # 获取GPU列表
    # 我们将在训练代码实现中初始化网络
    train_mul_gpu(net, num_gpus=2, batch_size=512, lr=0.1)


if __name__ == '__main__':
    train_rnnmodel_main()
    # train_cnnmodel_main()
