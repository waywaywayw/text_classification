# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-19
"""

import numpy as np
import math
import random


class Batcher(object):
    def __init__(self, data_X, data_Y=None, batch_size=32):
        """
        将数据制作成batch形式，并通过yield返回。

        :param data_X: 数据部分
        :param data_Y: 标签部分
        """
        self.data_X = data_X
        self.data_Y = data_Y
        self.batch_size = batch_size
        # self.shuffle()
        # 计算可以切分出多少个 batch
        self.batch_num = int(math.ceil(self.data_X.shape[0] / batch_size))  # 向上取整。最后一个batch会不够batch_size条数据。但是这样也没报错？
        # self.batch_num = int(math.floor(self.data_X.shape[0] / batch_size))  # 向下取整。保证每个batch都有batch_size条数据

    def next_batch(self):
        """
        将数据集切分成若干个 batch，并用 yield 逐个全部返回
        :return: 下一个batch的数据
        """
        batch_size = self.batch_size
        # 返回下一个batch的数据
        for idx in range(self.batch_num):
            # 切分数据x
            x_batch = self.data_X[batch_size * idx: batch_size * (idx + 1)]
            if self.data_Y is not None:
                # 切分标签y
                y_batch = self.data_Y[batch_size * idx: batch_size * (idx + 1)]
                yield x_batch, y_batch
            else:
                yield x_batch

    def random_batch(self, batch_num):
        """
        随机返回 batch_num 组数据 （每组 batch_size 个）
        """
        batch_size = self.batch_size
        for _ in range(batch_num):
            # 获取随机下标
            rand_idx = random.sample(range(0, self.data_size), batch_size)
            # 根据下标，获取数据子集
            x_batch = self.data_X[rand_idx]
            if self.data_Y is not None:
                y_batch = self.data_Y[rand_idx]
                yield x_batch, y_batch
            else:
                yield x_batch

    def shuffle(self):
        """将 data_X 和 data_Y 乱序"""
        perm = np.random.permutation(self.data_X.shape[0])
        self.data_X = self.data_X[perm]
        self.data_Y = self.data_Y[perm]

    @property
    def data_size(self):
        return len(self.data_X)