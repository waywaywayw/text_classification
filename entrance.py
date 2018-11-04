# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-10
"""

import os
from pprint import pprint
import tensorflow as tf

from mode_train import mode_train
from mode_eval import mode_evaluate
from mode_pred import mode_predict
from classifier_models.a06_fasttext import model_fasttext
from config.common import current_model_name, current_model_config

FLAGS = tf.app.flags.FLAGS
# 输入数据存放路径
tf.app.flags.DEFINE_string('input_path', os.path.join("input_data", "cantonese_data"), '输入数据存放路径.') # 自带的粤语句子数据
# tf.app.flags.DEFINE_string('input_path', os.path.join("input_data", "dbpedia_data"), '输入数据存放路径.')   # 自带的PTB数据
# tf.app.flags.DEFINE_string('input_path', os.path.join("input_data", "my_data"), '输入数据存放路径.') # 自己的数据集
# 执行模式
tf.app.flags.DEFINE_string('mode', 'train', '执行模式')   # train, eval, pred
# 选择分类模型的序号
# 1. AttnBiLSTM   2. IndRNN     3. HAN      4. Transformer      5.       6. fasttext(需要指定环境)
tf.app.flags.DEFINE_integer('model_idx', 2, '分类模型的序号')
# 是否开启测试config（数据规模变小）
tf.app.flags.DEFINE_boolean('small_flag', False, '是否调小数据集')
# tf.app.flags.DEFINE_boolean('small_flag', True, '是否调小数据集')


def small_config(config, small_flag):
    """将config的规模调小，方便快速调试
    """
    if small_flag:
        config.train_epoch = 3
        config.data_sample_ratio = 0.1
        config.predict_data_sample_ratio = 0.1
    return config


def main(unused_argv):
    # from config.a02_ind_rnn import IndRNN_config
    # config = IndRNN_config()
    # pprint(config)
    input_path = FLAGS.input_path
    mode = FLAGS.mode
    model_idx = FLAGS.model_idx

    # 选择配置文件
    if model_idx >= 1 and model_idx <= len(current_model_config):    # 确保 model_idx 合法
        model_name = current_model_name[model_idx]  # 根据序号，确定模型名字
        config = current_model_config[model_name]   # 根据模型名字，确定对应配置文件
        print('选择了第{}种模型: {}'.format(model_idx, model_name))
        print(config)   # ?
    else:
        raise TypeError("没有第{}种模型！".format(model_idx))

    # 是否开启测试config
    small_flag = FLAGS.small_flag
    config = small_config(config, small_flag)

    # 执行选定的模式
    print('选择了{}模式\n'.format(mode))
    if model_idx==6:    # 对fasttext模型特殊处理
        model_fasttext(mode, config, input_path)
    else :  # 处理其他模型
        if mode == "train":
            mode_train(config, input_path)
        elif mode == "eval":
            mode_evaluate(config, input_path)
        elif mode == "pred":
            mode_predict(config, input_path)
        else:
            raise TypeError("没有此模式：{}\n模式共有三种：train, eval, pred\n".format(mode))


if __name__ == "__main__":
    tf.app.run()
