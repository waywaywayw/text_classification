# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-25
"""

import os
import time
import copy
from sklearn import metrics
import tensorflow as tf

from utils.batcher import Batcher
from utils.data_helper import (
    load_data,
    make_vocabulary,
    load_vocabulary,
    data_preprocessing,
)
from utils.model_helper import (
    run_eval_step,
    run_one_epoch,
    metrics_model
)
from utils.mode_helper import (
    choose_model_by_name,
    get_and_check_ckpt_state
)


def evaluate(classifier, config, data_batcher):
    """
    开启session, 载入训练好的模型，真正执行验证步骤。
    对测试集数据进行分类。

    :param classifier: 分类器
    :param config: 配置文件
    :param test_batcher: 测试集数据  格式：list of documents word index  形如：[[256, 21, 58, 27 ...], [87, 948, 24, 0, 0 ...], ...]
    :return: 无。只得到测试集的测试结果。
    """
    train_dir = os.path.join(config.log_root, config.model_name, "train")
    eval_dir = os.path.join(config.log_root, config.model_name, "eval")
    summary_writer_eval = tf.summary.FileWriter(os.path.join(eval_dir, "summaries"))
    # 载入并验证训练好的模型是否存在
    ckpt_state = get_and_check_ckpt_state(train_dir)

    # 开启session
    saver = tf.train.Saver()
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    with tf.Session(config=config_proto) as sess:
        # 载入已训练的模型参数
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        # 评估数据
        return_dict = run_one_epoch(classifier, sess, config, run_eval_step, data_batcher, summary_writer_eval)
        print("Test  accuracy: %.3f %%, loss: %.4f" % (return_dict['acc'] * 100, return_dict['loss']))
        # 计算一些评价指标
        metrics_model(return_dict['real_label'], return_dict['predict'])


def mode_evaluate(config, input_path):
    """
    执行eval模式。评估模型。

    :param config: 配置文件
    :param input_path: 数据集路径
    :return: 无
    """
    # 读入数据
    x_test, y_test = load_data(
        os.path.join(input_path, "data_test.txt"),
        sample_ratio=config.data_sample_ratio,
        n_class=config.n_class,
        one_hot=config.one_hot,
    )
    print("成功载入测试集文件")
    # 读取已有字典
    my_vocab = load_vocabulary(max_vocab_size=config.max_vocab_size)
    config.vocab_size = my_vocab.vocab_size
    print("载入已有字典, 字典实际大小：{} ， 字典设置大小: {}".format(
        len(my_vocab.word_index) + 1, config.vocab_size
    ))

    # 数据预处理（转化为id表示，并padding)
    x_test = data_preprocessing(x_test, my_vocab, max_len=config.max_len)
    print("Test  Set size: %d" % len(x_test))

    config.keep_prob = 1.0
    # 创建分类器
    classifier = choose_model_by_name(config)
    classifier.build_graph()

    # 创建测试集的batcher
    test_batcher = Batcher(x_test, y_test, batch_size=config.batch_size)
    # 开始评估模型
    evaluate(classifier, config, test_batcher)
