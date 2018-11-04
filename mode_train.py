# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-25
"""

import os
import time
import copy
import tensorflow as tf

from utils.batcher import Batcher
from utils.data_helper import (
    load_data,
    make_vocabulary,
    data_preprocessing,
    split_dataset,
)
from utils.model_helper import (
    run_train_step,
    run_eval_step,
    run_some_steps,
    run_one_epoch,
    metrics_model
)
from utils.early_stopping import EarlyStopping
from utils.mode_helper import choose_model_by_name


def train(classifier, config, train_batcher, valid_batcher, test_batcher):
    """
    开启session, 真正的执行训练步骤。
    每训练完一个epoch之后使用验证集数据验证模型，并写入summary。
    指定的epoch数（配置文件的train_epoch项）训练完后，使用测试集数据测试模型。

    :param classifier: 分类器
    :param config: 配置文件
    :param train_batcher: 训练集数据的batch生成器。
                            batch的shape：[config.batch_size, config.max_len]
    :param valid_batcher: 验证集数据的batch生成器。
    :param test_batcher: 测试集数据的batch生成器。
    :return: 无。模型的训练进度保存在路径：config.log_root/config.model_name/train
    """
    # 创建eval_config，与config唯一不同是 dropout设置为1.0；验证和测试模型时使用eval_config
    eval_config = copy.deepcopy(config)
    eval_config.keep_prob = 1.0
    # 定义模型训练进度和summary的存放地址 train_dir
    train_dir = os.path.join(config.log_root, config.model_name, "train")
    # 定义训练集和验证集的summary_writer
    summary_writer_train = tf.summary.FileWriter(
        os.path.join(train_dir, "summaries", "summaries_train")
    )
    summary_writer_valid = tf.summary.FileWriter(
        os.path.join(train_dir, "summaries", "summaries_valid")
    )
    saver = tf.train.Saver(max_to_keep=3)
    early_stop = EarlyStopping(config.patience, mode='min')     # 设置early_stop. 监控元素为 验证集的loss
    # 配置参数：内存自增长
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    # 创建MonitoredSession
    with tf.train.MonitoredTrainingSession(
        is_chief=True,
        checkpoint_dir=train_dir,
        scaffold=None,  # ?
        hooks=None,  # ?
        chief_only_hooks=None,  # ?
        save_checkpoint_secs=None,  # config.
        save_summaries_steps=None,
        save_summaries_secs=None,
        config=config_proto,
    ) as sess:
        # 计算需要运行的step数
        if config.train_epoch:
            step_num = int(config.train_epoch * train_batcher.data_size / train_batcher.batch_size)
        else:
            step_num = config.train_step
        start = time.time()
        for step in range(step_num):
            if sess.should_stop():  # 可能有未知情况需要停止
                break
            # run train data one epoch
            t0 = time.time()
            global_step = classifier.global_step.eval(sess)
            print("\nglobal_step : {}".format(global_step))
            return_dict = run_some_steps(classifier, sess, config, run_train_step, train_batcher, summary_writer_train, step_num=config.once_train_step)
            print("Train time: %.3f s" % (time.time()-t0))
            print("Train accuracy: %.3f %%, loss: %.4f" % (return_dict['acc'] * 100, return_dict['loss']))
            # evaluate valid data
            tf.logging.set_verbosity(tf.logging.ERROR)
            return_dict = run_one_epoch(classifier, sess, config, run_eval_step, valid_batcher, summary_writer_valid)
            tf.logging.set_verbosity(tf.logging.WARN)
            print("Valid accuracy: %.3f %%, loss: %.4f" % (return_dict['acc'] * 100, return_dict['loss']))
            # early stopping
            if early_stop.add_monitor(return_dict['loss']):
                # 保存
                print('发现更优的模型参数，进行保存..')
                # 这个奇葩的写法，详见：https://github.com/tensorflow/tensorflow/issues/8425
                saver.save(sess._sess._sess._sess._sess, os.path.join(train_dir, 'model.ckpt'), global_step=global_step)
            if early_stop.stop():
                print('触发early stopping! 前 {} 个 monitor 为 {}'.format(early_stop.patience, early_stop.pre_monitor))
                break
        print("Training finished, time consumed : %.3f s" % (time.time() - start))

        # evaluate test data
        print("\nStart evaluating:")
        tf.logging.set_verbosity(tf.logging.ERROR)
        return_dict = run_one_epoch(classifier, sess, config, run_eval_step, test_batcher)
        tf.logging.set_verbosity(tf.logging.WARN)
        print("Test  accuracy: %.3f %%, loss: %.4f" % (return_dict['acc'] * 100, return_dict['loss']))
        # 计算评价指标
        metrics_model(return_dict['real_label'], return_dict['predict'])

def mode_train(config, input_path):
    """
    执行train模式。按照给定配置，训练模型。

    :param config: 配置文件
    :param input_path: 数据集路径
    :return: 无
    """
    # 读入训练集和测试集
    x_train, y_train = load_data(
        os.path.join(input_path, "data_train.txt"),
        sample_ratio=config.data_sample_ratio,
        n_class=config.n_class,
        one_hot=config.one_hot,
    )
    print("成功载入训练集文件")
    x_test, y_test = load_data(
        os.path.join(input_path, "data_test.txt"),
        sample_ratio=config.data_sample_ratio,
        n_class=config.n_class,
        one_hot=config.one_hot,
    )
    print("成功载入测试集文件")
    # 获取验证集
    if os.path.isfile(os.path.join(input_path, "data_valid.txt")):
        # 从验证集文件中获取
        x_valid, y_valid = load_data(
            os.path.join(input_path, "data_test.txt"),
            sample_ratio=config.data_sample_ratio,
            n_class=config.n_class,
            one_hot=config.one_hot,
        )
        print("成功载入验证集文件")
    else:
        # 将测试集的一部分分割出来，作为验证集
        split_radio = config.valid_test_split_radio  # 设置分割比例
        x_test, x_valid, y_test, y_valid = split_dataset(x_test, y_test, split_radio)
        print("没有发现验证集文件，已分割测试集的 {}% 来作为验证集".format(split_radio*100))

    # 创建字典
    my_vocab = make_vocabulary(x_train, max_vocab_size=config.max_vocab_size)
    config.vocab_size = my_vocab.vocab_size
    print("使用训练集数据 制作字典完成, 字典实际大小：{} ， 字典设置大小: {}".format(
            len(my_vocab.word_index) + 1, config.vocab_size
    ))

    # 数据预处理(转化为id表示，并padding)
    print('开始对数据集进行预处理 (word表示 -> id表示)')
    x_train = data_preprocessing(x_train, my_vocab, max_len=config.max_len)
    x_valid = data_preprocessing(x_valid, my_vocab, max_len=config.max_len)
    x_test = data_preprocessing(x_test, my_vocab, max_len=config.max_len)
    print("Train Set size: %d" % len(x_train))
    print("Valid Set size: %d" % len(x_valid))
    print("Test  Set size: %d" % len(x_test))

    # 创建分类器
    classifier = choose_model_by_name(config)
    classifier.build_graph()

    # 创建训练集、验证集、测试集的 batcher
    train_batcher = Batcher(x_train, y_train, batch_size=config.batch_size)
    valid_batcher = Batcher(x_valid, y_valid, batch_size=config.batch_size)
    test_batcher = Batcher(x_test, y_test, batch_size=config.batch_size)
    # 开始训练模型
    train(classifier, config, train_batcher, valid_batcher, test_batcher)
