# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018/10/8
"""
import os
from config.classifier import classifier_config


class fasttext_config(classifier_config):
    model_name = "fasttext"

    # train parameter
    train_epoch = 100
    learning_rate = 1e-1
    dim = 100
    wordNgrams = 2
    loss = 'softmax'
    verbose = 1     # ?
    # maxn = 2

    # 以下为重要参数！
    # 设置是否是纯python版本形式安装的fasttext！True代表纯python版本形式安装
    is_python_package = True
    # fasttext的实际运行路径. 比如运行fasttext需要输入 /home/fasttext 那么设置fasttext_path = "/home/"
    # 仅在facebook版本形式下才需要设置该参数
    fasttext_path = ""




