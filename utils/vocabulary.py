# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-17
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import OrderedDict
from hashlib import md5
import string
import sys

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin


def text_to_word_sequence(
    text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" "
):
    """
    用split字符 分割 text 成 word list，并且过滤掉所有filters字符

    :param text: Input text (string).
    :param filters: Sequence of characters to filter out.
    :param lower: Whether to convert the input to lowercase.
    :param split: Sentence split marker (string).
    :return: A list of words (or tokens).
    """
    if lower:  # 转换为小写
        text = text.lower()
    # 生成filters字符的映射
    translate_map = str.maketrans(filters, split * len(filters))
    # 过滤掉filters字符
    text = text.translate(translate_map)
    # 分割 text 成 word list
    seq = text.split(split)
    # 返回 word list (去掉None项)
    return [i for i in seq if i]


class MyVocabulary(object):
    """
    字典类。实现时参照了tf.keras.preprocessing.text.Tokenizer，删掉了其中对py2的兼容代码。
    提供功能：
        从文章列表中获取字典(fit_on_texts)；从字典文件中获取字典(fit_on_vocab_file)。
        将文章列表从word表示转换为id表示(texts_to_sequences)。
        保存MyVocabulary中的字典信息到指定文件(save_vocab)。
    """
    def __init__(
        self,
        num_words=None,
        filters='？！。，、!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ",
        char_level=False,
        oov_token=None,
    ):
        """
        :param num_words: 限制词典的大小
        :param filters: 需要过滤掉的字符
        :param lower: 是否需要进行小写转换
        :param split: 用于分割text的分割符
        :param char_level: 是否进行char级的分割（不再使用split分隔符来分割）
        :param oov_token: OOV词对应的标识字符串
        """
        self.word_counts = OrderedDict()  # 词频表。  形式：单词 词频
        self.word_docs = {}  # 单词在文档中出现的次数。  形式：单词 包含该单词的文档数
        self.filters = filters  #
        self.split = split
        self.lower = lower
        self.num_words = num_words  # 手动限制的最大词典数
        self.document_count = 0
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = {}
        self.vocab_size = 0  # 真正的词典数

    def fit_on_vocab_file(self, vocab_path):
        """
        从已存在的字典文件中读取单词数据。
        格式：单词 频数
        运行完毕后，会构建好两个量：self.word_index（可能带OOV词) 和 self.word_counts（不带OOV词）
        """
        self.word_index = {}
        with open(vocab_path, "r", encoding="utf8") as vocab_f:
            # 保留idx=0。用来padding
            current_idx = 1
            for line in vocab_f:
                # pieces[0]是单词, pieces[1]是词频
                pieces = line.split()
                # 两个异常处理
                if len(pieces) != 2:
                    sys.stderr.write("Bad line: %s\n" % line)
                    continue
                if pieces[0] in self.word_index:
                    raise ValueError("Duplicated word: %s." % pieces[0])
                # 存入word_to_id
                self.word_index[pieces[0]] = current_idx
                self.word_counts[pieces[0]] = int(pieces[1])
                current_idx += 1
        # 添加oov_token
        if self.oov_token is not None:
            i = self.word_index.get(self.oov_token)
            if i is None:
                self.word_index[self.oov_token] = len(self.word_index) + 1

        # 计算词典的真正大小
        self.vocab_size = len(self.word_index) + 1  # 因为还有padding的下标0
        if (
            self.vocab_size > self.num_words
        ):  # 统计的词典数 和 手动设置的最大词典数，取较小值作为真正的词典数self.vocab_size
            self.vocab_size = self.num_words

    def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.
        word_index从下标1开始映射单词。0不用，留给padding。
        如果设置了oov_token, 则在word_index最末尾添加一个下标映射oov_token

        In the case where texts contains lists, we assume each entry of the lists
        to be a token.

        Required before using `texts_to_sequences` or `texts_to_matrix`.

        Arguments:
            texts: can be a list of strings,
                a generator of strings (for memory-efficiency),
                or a list of list of strings.
        """
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                seq = text
            else:
                seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))
        )

        if self.oov_token is not None:
            i = self.word_index.get(self.oov_token)
            if i is None:
                self.word_index[self.oov_token] = len(self.word_index) + 1

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

        # 计算词典的大小
        self.vocab_size = len(self.word_index) + 1  # 因为还有padding的下标0
        if (
            self.vocab_size > self.num_words
        ):  # 统计的词典数 和 手动设置的最大词典数，取较小值作为真正的词典数self.vocab_size
            self.vocab_size = self.num_words

    def texts_to_sequences(self, texts):
        """Transforms each text in texts in a sequence of integers.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        Arguments:
            texts: A list of texts (strings).

        Returns:
            A list of sequences.
        """
        res = []
        for vect in self.texts_to_sequences_generator(texts):
            res.append(vect)
        return res

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` in a sequence of integers.

        Each item in texts can also be a list, in which case we assume each item of
        that list
        to be a token.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        Arguments:
            texts: A list of texts (strings).

        Yields:
            Yields individual sequences.
        """
        for text in texts:
            if self.char_level or isinstance(text, list):  # char级 分割
                seq = text
            else:  # 使用split分隔符 分割
                seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if (
                        self.num_words and i >= self.num_words
                    ):  # 如果遇到 在词典里但是超过num_words 的词，那么直接跳过该词！
                        continue
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    # 如果遇到OOV词，那么替换为oov_token！
                    # 如果oov_token不在词典里（一般不会有这个情况，因为都会设置oov_token），那么直接跳过该词！
                    i = self.word_index.get(self.oov_token)
                    if i is not None:
                        vect.append(i)
            yield vect

    def save_vocab(self, save_path, encoding="utf8"):
        """
        保存词典到指定路径。词典只保存：单词 词频
        :param save_path: 保存路径
        """
        with open(save_path, "w", encoding=encoding) as fout:
            # 先按照词频排序
            wcounts = list(self.word_counts.items())
            wcounts.sort(key=lambda x: x[1], reverse=True)
            # 再保存
            for (word, cnt) in wcounts:
                fout.write("{} {}\n".format(word, cnt))

    def add_on_texts(self):
        """
        增加新的语料(text of list)。 扩充word_index, 并且需要确保新语料的词出现在num_words之内。 to be continue..
        """
        pass

    def get_word2index(self):
        """返回word2index表"""
        return self.word_index[: self.num_words]

    def get_index2word(self):
        """返回index2word表"""
        w2i = self.get_word2index()
        i2w = {}
        for word, idx in w2i.items():
            i2w[idx] = word
        return i2w
