# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-20
"""
import os


class classifier_config(object):
    """
    创建每个配置文件都通用的设置项
    """
    # 训练进度 和 日志summary的存放路径
    log_root = os.path.join("classifier_log")
    # 验证集和测试集的划分比例。设置为0.1 ，表示 验证集：测试集 = 1：9
    valid_test_split_radio = 0.1
    # 预测集抽样率（默认1.0）
    predict_data_sample_ratio = 1.0
    # 隔多久输出训练并计算验证集
    once_train_step = 100
    # early stopping. 验证集的loss多少次没有下降了就结束训练
    patience = 30

    def __repr__(self):
        str = 'RNNCell={}, learning_rate={}, keep_prob={}'.format(self.RNNCell, self.learning_rate, self.keep_prob)
        return str