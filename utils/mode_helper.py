# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-25
"""


import time
import copy
import numpy as np
import tensorflow as tf

from utils.batcher import Batcher
from utils.data_helper import (
    load_data,
    load_predict_data,
    make_vocabulary,
    load_vocabulary,
    data_preprocessing,
    split_dataset,
)
from utils.model_helper import (
    run_train_step,
    run_eval_step,
    run_predict_step,
    run_some_steps
)
from config.common import current_model


def get_and_check_ckpt_state(train_dir):
    """
    获取最新的训练进度，并检查是否合法。
    :param train_dir: 保存训练进度的路径
    :return: 最新训练进度checkpoint
    """
    try:
        ckpt_state = tf.train.get_checkpoint_state(train_dir)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            return IOError("No model to eval yet at %s", train_dir)
    except tf.errors.OutOfRangeError as e:
        return IOError("Cannot restore checkpoint: %s", e)
    return ckpt_state


def choose_model_by_name(config):
    """
    根据配置文件中的模型名字，创建对应的模型。
    现有的模型列表存储在：配置文件classifier_config/classifier.py 中的 current_model 项

    :param config: 配置文件
    :return: 创建好的模型
    """
    for (model_name, model) in current_model.items():
        if config.model_name == model_name:
            return model(config)
    raise TypeError("没有找到指定的模型：{}\n目前的模型共有{}种：{}".format(
            config.model_name, len(current_model), current_model.keys()))


