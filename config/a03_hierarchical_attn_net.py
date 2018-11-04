# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018/10/8
"""
import os
from config.classifier import classifier_config


class HAN_config(classifier_config):
    model_name = "hierarchical_attn_lstm"

    # model parameter
    batch_size = 1024   # ？又那么大
    attention_size = 64
    max_len = 256
    hidden_size = 64
    vocab_size = -1  # 根据实际词典进行赋值
    max_vocab_size = 10000  # 限制最大词典数
    embedding_size = 128
    # LAMBDA = 0.0001     # ？
    n_class = 15
    RNNCell = "BLOCK"  # BASIC, BLOCK
    one_hot = True

    # train parameter
    learning_rate = 1e-3
    keep_prob = 0.9
    train_epoch = 10
    checkpoint_secs = 60
    data_sample_ratio = 1.0  # 1.0, 0.5, 0.2, 1e-1, 1e-2

    # eval parameter
    # predict parameter
