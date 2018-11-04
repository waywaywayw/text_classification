# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-10
"""

import numpy as np
from sklearn import metrics
import tensorflow as tf


def run_train_step(model, sess, config, batch):
    """
    执行一轮train step。
    返回summaries, acc, loss, global_step
    """
    feed_dict = {
        model.x: batch[0],
        model.label: batch[1],
        model.keep_prob: config.keep_prob,
    }
    to_return = {
        'train_op': model.train_op,
        'summaries': model.summaries,
        'acc': model.accuracy,
        'loss': model.loss,
        'global_step': model.global_step,
    }
    return sess.run(to_return, feed_dict)


def run_eval_step(model, sess, config, batch):
    """
    执行一轮eval step。
    返回summaries, acc, loss, global_step
    """
    feed_dict = {
        model.x: batch[0],
        model.label: batch[1],
        model.keep_prob: config.keep_prob,
    }
    to_return = {
        'summaries': model.summaries,
        'acc': model.accuracy,
        'loss': model.loss,
        'real_label': model.real_label,     # 取出真实标签real_label和预测标签predict，来计算更多评价指标
        'predict': model.predict,           # 取出真实标签real_label和预测标签predict，来计算更多评价指标
        'global_step': model.global_step,
    }
    return_dict = sess.run(to_return, feed_dict)
    return return_dict


def run_predict_step(model, sess, config, batch):
    """
     执行一轮predict step。
     返回预测值predict, 预测值概率predict_prob, global_step
     """
    feed_dict = {
        model.x: batch,     # 这里因为没有标签y了，所以不是batch[0]了
        model.keep_prob: config.keep_prob
    }
    to_return = {
        'predict': model.prediction,
        'predict_prob': model.predict_prob,
        'global_step': model.global_step,
    }
    return_dict = sess.run(to_return, feed_dict)
    # 获取预测类别的概率
    return_dict['predict_prob'] = [ pred_prob_list[pred] for pred, pred_prob_list in zip(return_dict['predict'], return_dict['predict_prob'])]
    return return_dict


def run_some_steps(model, sess, config, run_step, data_batcher, summary_writer=None, step_num=None):
    """
    运行一轮epoch。
    同时将run_step方法返回的数据收集起来，统一递交给上层方法。

    :param model: 当前模型
    :param sess: 当前会话
    :param config: 配置文件
    :param run_step: 需要运行的step
    :param data_batcher: 数据迭代器
    :param step_num: 需要执行的step数。如果为None,那么就是整个epoch
    :param summary_writer: summary写入器。为None时不需写入summary
    :return: return_dict: 该轮epoch收集的所有数据
    """
    loss_list = []  # 记录每轮step得到的loss
    acc_list = []   # 记录每轮step得到的acc
    predict_list = []   # 记录每个数据的预测标签
    predict_prob_list = []  # 记录每个数据的预测概率
    real_label_list = []    # 记录每个数据的真实标签
    if step_num:
        batch_genor = data_batcher.random_batch(step_num)
    else:
        batch_genor = data_batcher.next_batch()
    for x_batch, y_batch in batch_genor:
        # 运行给定的step
        return_dict = run_step(
            model, sess, config, (x_batch, y_batch)
        )
        # 根据返回的return_dict, 追加需要的项
        if 'loss' in return_dict:
            loss_list.append(return_dict['loss'])
        if 'acc' in return_dict:
            acc_list.append(return_dict['acc'])
        if 'predict' in return_dict:
            predict_list.extend(return_dict['predict'])
        if 'predict_prob' in return_dict:
            predict_prob_list.extend(return_dict['predict_prob'])
        if 'real_label' in return_dict:
            real_label_list.extend(return_dict['real_label'])

        # 运行完一个step, 将训练结果写入summary
        if summary_writer is not None:
            summary_writer.add_summary(return_dict['summaries'], return_dict['global_step'])
    # 运行完一轮epoch, 将summary推送为最新状态
    if summary_writer is not None:
        summary_writer.flush()

    # 整理需要返回的项
    return_dict = {}
    if loss_list is not None:
        return_dict['loss'] = np.asarray(loss_list).mean()  # 统计均值
    if acc_list is not None:
        return_dict['acc'] = np.asarray(acc_list).mean()
    if predict_list is not None:
        return_dict['predict'] = predict_list
    if predict_prob_list is not None:
        return_dict['predict_prob'] = predict_prob_list
    if real_label_list is not None:
        return_dict['real_label'] = real_label_list
    return return_dict


def run_one_epoch(model, sess, config, run_step, data_batcher, summary_writer=None):
    return run_some_steps(model, sess, config, run_step, data_batcher, summary_writer=summary_writer)


def metrics_model(real_label, predict):
    """根据真实值和预测值，计算模型的一些评估指标
    """
    # 直接使用sklearn的分类器报告..
    FLAGS = tf.app.flags.FLAGS
    target_names = None
    if str(FLAGS.input_path) == 'input_data\cantonese_data' :
        target_names = ['Chinese', 'Cantonese']
    print(metrics.classification_report(real_label, predict, target_names=target_names, digits=4))