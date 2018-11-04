# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-25
"""


import os
import numpy as np
import tensorflow as tf

from utils.batcher import Batcher
from utils.data_helper import (
    load_predict_data,
    load_vocabulary,
    data_preprocessing
)
from utils.model_helper import (
    run_predict_step,
    run_one_epoch
)
from utils.mode_helper import (
    choose_model_by_name,
    get_and_check_ckpt_state
)

# 预测值的保存路径
output_path = os.path.join("output", "predict_result.txt")


def predict(classifier, config, data_batcher):
    """
    开启session, 载入训练好的模型，真正执行预测步骤。
    对待预测数据进行预测。

    :param classifier: 分类器
    :param config: 配置文件
    :param data_batcher: 数据集  格式：list of documents word index  形如：[[256, 21, 58, 27 ...], [87, 948, 24, 0, 0 ...], ...]
    :return: 对每个数据的预测值   格式：list of predict value  形如：[1, 2, 0, 3, 2, 1, ...]
    """
    train_dir = os.path.join(config.log_root, config.model_name, "train")
    # 获取并验证已训练的模型是否存在
    ckpt_state = get_and_check_ckpt_state(train_dir)

    # 开启session
    saver = tf.train.Saver()
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    with tf.Session(config=config_proto) as sess:
        # 载入已训练的模型参数
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        # 预测数据
        return_dict = run_one_epoch(classifier, sess, config, run_predict_step, data_batcher)
        # 将预测值和预测概率 整合到一个列表
        predict_list = [(pred, pred_prob) for (pred, pred_prob) in zip(return_dict['predict'], return_dict['predict_prob'])]
    return predict_list


def mode_predict(
    config, input_path, output_path=output_path
):
    """
    执行predict模式。对给定数据 进行预测。

    :param config: 配置文件
    :param input_path: 待预测 数据集路径
    :param output_path: 预测结果 保存路径
    :return: 无。预测值已写入指定文件
    """
    # 读入数据
    x_test = load_predict_data(
        os.path.join(input_path, "data_predict.txt"),
        sample_ratio=config.predict_data_sample_ratio
    )
    print("成功载入待预测文件")
    # 读取已有字典
    my_vocab = load_vocabulary(max_vocab_size=config.max_vocab_size)
    config.vocab_size = my_vocab.vocab_size
    print("载入已有字典, 字典实际大小：{} ， 字典设置大小: {}".format(
        len(my_vocab.word_index) + 1, config.vocab_size
    ))

    # 数据预处理（转化为id表示，并padding)
    x_test = data_preprocessing(x_test, my_vocab, max_len=config.max_len)
    print("Data Set size: %d" % len(x_test))

    config.keep_prob = 1.0
    # 创建分类器
    classifier = choose_model_by_name(config)
    classifier.build_graph()

    # 创建数据集的batcher
    data_batcher = Batcher(x_test, batch_size=config.batch_size)
    # 开始预测数据
    print('开始预测数据')
    predict_list = predict(classifier, config, data_batcher)

    # 保存预测值到output文件夹
    with open(output_path, "w", encoding="utf8") as fout:
        for (pred, pred_prob) in predict_list:
            fout.write("%d\t%f\n" % (pred, pred_prob))
    print('预测完成，并已将预测值写入输出文件：', output_path)
