# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-10
"""
import os
from config.classifier import classifier_config


class AttnBiLSTM_config(classifier_config):
    model_name = "attn_bi_lstm"

    # model parameter
    batch_size = 64
    max_len = 100
    hidden_size = 64
    vocab_size = -1  # 根据实际词典进行赋值
    max_vocab_size = 10000  # 限制最大词典数
    embedding_size = 128
    n_class = 2
    RNNCell = "BLOCK"  # BASIC, BLOCK, GRU
    one_hot = False

    # train parameter
    learning_rate = 1e-3
    keep_prob = 0.9
    train_epoch = 50
    checkpoint_secs = 60
    data_sample_ratio = 1.0  # 1.0, 0.5, 0.2, 1e-1, 1e-2

    # eval parameter
    # predict parameter

    def __repr__(self):
        config = AttnBiLSTM_config
        ret = 'batch_size = {}\n'.format(config.batch_size) + \
              'RNNCell = {}\n'.format(config.RNNCell) + \
              'learning_rate = {}\n'.format(config.RNNCell) + \
              'keep_prob = {}\n'.format(config.RNNCell) + \
              'train_epoch = {}\n'.format(config.RNNCell) + \
              'data_sample_ratio = {}'.format(config.data_sample_ratio)
        return ret


class AttnBiLSTM_common_config(classifier_config):
    model_name = "AttnBiLSTM"

    # model parameter
    batch_size = 32
    max_len = 100
    hidden_size = 64
    vocab_size = -1  # 根据实际词典进行赋值
    max_vocab_size = 20000  # 限制最大词典数
    embedding_size = 128
    n_class = 15
    RNNCell = "BLOCK"  # BASIC, BLOCK
    one_hot = False

    # train parameter
    learning_rate = 1e-3
    keep_prob = 0.5
    train_epoch = 20
    checkpoint_secs = 60
    data_sample_ratio = 1.0  # 1.0, 0.5, 0.2, 1e-1, 1e-2

    # eval parameter
    # predict parameter
