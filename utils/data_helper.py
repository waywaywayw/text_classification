# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-10
"""

import random
import os
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from utils.vocabulary import MyVocabulary


# 字典存放路径。   # train模式将会创建字典；eval, predict模式将会读取字典
vocab_folder = os.path.join("output", "vocab_file")
vocab_path = os.path.join("output", "vocab_file", "vocab.txt")


def load_data(
    input_path, sample_ratio=1.0, n_class=0, one_hot=False, shuffle=True, encoding="utf8"
):
    """
    从形如fasttext输入格式的纯文本文件中读入数据。
    一行是一条完整的数据，包括标签 和 空格分隔开的单词列表。 fasttext的输入格式 形如：__label__1 word1 word2 word3 ...
    注意：只能有一个类别标签，且标签必须为正整数！

    :param input_path: 输入文件的路径
    :param sample_ratio: 数据采样率。默认使用100%的数据
    :param n_class: 标签类别数目。如果指定使用one_hot形式输出，需要指定该项的值
    :param one_hot: 标签y是否使用one_hot形式输出。默认为False
    :return: 数据x 和 标签y . 数据x 形如：[[what are you doing?], [how are you!] ...] ; 标签y 形如：[ 1, 2, 0, 3, 1 ...]
    """
    x = []  # 数据x
    y = []  # 标签y
    with open(input_path, "r", encoding=encoding) as fin:
        fin_lines = fin.readlines()
        if shuffle:
            random.seed(2018)
            fin_lines = random.sample(fin_lines, int(len(fin_lines) * sample_ratio))  # 乱序
        else:
            fin_lines = fin_lines[: int(len(fin_lines) * sample_ratio)]    # 非乱序
        for line_idx, line in enumerate(fin_lines):
            words = line.split()
            # word[0] == '__label__xxx' xxx是整数
            y.append(int(words[0][9:]))
            x.append(" ".join(words[1:]))
    y = np.asarray(y).astype(int)
    if one_hot:  # 对标签y进行one_hot转换
        y = np.eye(n_class)[y]
    return x, y


def load_predict_data(input_path, sample_ratio=1.0, input_encoding="utf8"):
    """
    读入待预测数据。
    一行是一条完整的数据，包括空格分隔开的单词列表。 fasttext待预测数据的格式 形如：word1 word2 word3 ...

    :param input_path: 输入文件的路径
    :param sample_ratio: 数据采样率。默认使用100%的数据
    :return: 数据x. 数据x 形如：[[what are you doing?], [how are you!] ...]
    """
    # 提示是否载入了全部的待预测数据
    if sample_ratio<1.0 :
        print('提示：并没有载入全部待预测数据。检测到配置文件中data_sample_ratio项不为1')

    x = []  # 数据x
    with open(input_path, "r", encoding=input_encoding) as fin:
        fin_lines = fin.readlines()
        # fin_lines = random.sample(fin_lines, int(len(fin_lines) * sample_ratio))  # 乱序
        fin_lines = fin_lines[ : int(len(fin_lines) * sample_ratio)]    # 非乱序
        for line_idx, line in enumerate(fin_lines):
            x.append(line.strip())
    return x


def make_vocabulary(
    x_data,
    max_vocab_size = None,
    vocab_path=vocab_path
):
    """
    根据文本数据x_data 制作字典，并保存字典文件。

    :param x_data: 文本数据
    :param max_vocab_size: 手动限制最大词典数
    :return: 字典类对象
    """
    if not os.path.exists(vocab_folder):  # 如果不存在保存字典的文件夹，那么先创建
        os.makedirs(vocab_folder)

    # 创建字典类
    my_vocab = MyVocabulary(num_words=max_vocab_size, oov_token="<UNK>")
    # 分析x_data，得到字典信息
    my_vocab.fit_on_texts(x_data)
    # 保存字典到指定路径。 格式：单词 词频
    my_vocab.save_vocab(vocab_path)
    return my_vocab


def load_vocabulary(
    max_vocab_size=None,
    vocab_path=vocab_path
):
    """
    从字典文件中载入字典信息。

    :param max_vocab_size: 手动限制最大词典数
    :param vocab_path: 字典文件的路径
    :return: 字典类对象
    """
    # 合法性检查
    if not os.path.isfile(vocab_path) :
        raise TypeError('未能找到字典文件！请检查是否已执行过train模式')
    # 创建字典类
    my_vocab = MyVocabulary(num_words=max_vocab_size, oov_token="<UNK>")
    # 载入字典信息
    my_vocab.fit_on_vocab_file(vocab_path)
    return my_vocab


def data_preprocessing(x_data, vocab, max_len):
    """
    将数据x_data由word表示转化为id表示，并进行padding

    :param x_data: 数据样本。调用load_data方法得到
    :param vocab: 字典类对象
    :param max_len: 每个数据样本需要的长度。如果不够长，则进行padding；如果超长了，则截断。
    :return: 预处理好的数据样本
    """
    x_data_idx = vocab.texts_to_sequences(x_data)
    x_data_padded = pad_sequences(
        x_data_idx, maxlen=max_len, padding="post", truncating="post"
    )
    return x_data_padded


def split_dataset(x, y=None, split_ratio=0.2):
    """
    根据分割比例，把一份数据集分割成两份。

    :param x: 带分割的数据集，数据部分。
    :param y: 带分割的数据集，标签部分。可以没有标签部分
    :param split_ratio: 分割比例
    :return: 分割后的数据集
    """
    """split dataset to two sets with ratio """
    split_size = (int)(len(x) * split_ratio)
    x_1 = x[split_size:]
    x_2 = x[:split_size]
    if y is not None:
        y_1 = y[split_size:]
        y_2 = y[:split_size]
        return x_1, x_2, y_1, y_2
    else:
        return x_1, x_2


def merge_dataset():
    """
    合并数据集   to do
    :return:
    """
    pass
