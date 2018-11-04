# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-10-11
"""
import os
from config.classifier import classifier_config


# 粤语句子数据的实验参数
class transformer_config(classifier_config):
    model_name = "transformer"

    # model parameter
    batch_size = 32
    max_len = 32
    hidden_size = 64
    vocab_size = -1  # 根据实际词典进行赋值
    max_vocab_size = 10000  # 限制最大词典数
    embedding_size = 128
    n_class = 2
    RNNCell = "BLOCK"  # BASIC, BLOCK
    one_hot = False

    # train parameter
    learning_rate = 1e-3
    keep_prob = 0.5
    train_epoch = 30
    checkpoint_secs = 60
    data_sample_ratio = 0.5  # 1.0, 0.5, 0.2, 1e-1, 1e-2

    # eval parameter
    # predict parameter


