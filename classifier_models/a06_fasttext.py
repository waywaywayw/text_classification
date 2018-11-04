# -*- coding: utf-8 -*-
"""
fasttext和其他的模型都不同。因为fasttext是调用写好的api
@author: weijiawei
@date: 2018/10/08
"""

import os
import re
try:
    from fastText import FastText as fasttext  # facebook版本安装
    fasttext_path = ""  # 设置fasttext地址
except:
    import fasttext # 纯python版本安装


def pretrained_vectors(config, input_path):
    """预训练词向量
    """
    input_path = os.path.join(input_path, 'data_train.txt')
    output_path = os.path.join(config.log_root, config.model_name, "train", "word_vector")
    # 执行训练命令
    print('开始预训练词向量（仅需要训练一次。一般来说需要一些时间..） ')
    if config.is_python_package:
        model = fasttext.skipgram(input_file=input_path, output=output_path,
                          epoch=config.train_epoch, dim=config.dim)
        # print(model.words)
    else:
        train_command = '{}fasttext skipgram '.format(fasttext_path) + \
                        '-input {} -output {} '.format(input_path, output_path) + \
                        '-epoch {} -dim {} '.format(config.train_epoch, config.dim)
        os.system(train_command)
    print('预训练词向量完毕')


def mode_train_fasttext(config, _input_path):
    """训练fasttext模型并保存训练好的模型
    """
    input_path = os.path.join(_input_path, 'data_train.txt')
    output_path = os.path.join(config.log_root, config.model_name, "train", "ftmodel")
    pretrained_path = os.path.join(config.log_root, config.model_name, "train", "word_vector.vec")
    # 执行训练命令
    print('开始训练fasttext模型')
    if config.is_python_package:
        # python版必须指定bucket, 未修复的bug..  bucket增加多少倍，保存的模型大小大致也会增加多少倍
        fasttext.supervised(input_file=input_path, output=output_path,
                    epoch=config.train_epoch, lr=config.learning_rate, dim=config.dim, word_ngrams=config.wordNgrams,
                    loss=config.loss, pretrained_vectors=pretrained_path, bucket=200000)
    else:
        train_command = '{}fasttext supervised '.format(fasttext_path) + \
            '-input {} -output {} '.format(input_path, output_path) + \
            '-epoch {} -lr {} -dim {} -wordNgrams {} '.format(config.train_epoch, config.learning_rate, config.dim, config.wordNgrams)+ \
            '-loss {} -verbose {} -pretrainedVectors {}'.format(config.loss, config.verbose, pretrained_path)
            # '-loss {} -verbose {}'.format(config.loss, config.verbose)
        os.system(train_command)
    print('训练fasttext模型完毕')


def mode_evaluate_fasttext(config, input_path):
    """测试训练好的fasttext模型
    """
    input_path = os.path.join(input_path, 'data_test.txt')
    ftmodel_path = os.path.join(config.log_root, config.model_name, "train", "ftmodel.bin")
    # 执行测试命令
    if config.is_python_package:
        ftmodel = fasttext.load_model(ftmodel_path)
        test_log = ftmodel.test(input_path)
        print('测试集的测试结果:\nN\t{}\nP@1\t{}\nR@1\t{}\n'.format(test_log.nexamples, test_log.precision, test_log.recall))
    else:
        test_command = '{}fasttext test '.format(fasttext_path) + \
            '{} {}'.format(ftmodel_path, input_path)
        test_log = os.popen(test_command)
        print('测试集的测试结果:\n{}\n'.format(test_log.read()))


def mode_predict_fasttext(config, input_path):
    """使用训练好的fasttext模型预测 数据
    """
    input_path = os.path.join(input_path, 'data_predict.txt')
    ftmodel_path = os.path.join(config.log_root, config.model_name, "train", "ftmodel.bin")
    # 最终预测值的保存路径
    output_path = os.path.join("output", "predict_result.txt")

    # 载入训练好的模型
    ftmodel = fasttext.load_model(ftmodel_path)
    print('开始预测数据')
    with open(input_path, 'r', encoding='utf8') as fin:
        lines = fin.readlines()
        lines = [x.strip() for x in lines]  # 去掉每行头尾空白
        if config.is_python_package:
            # python版本只有 predict_proba 方法才能给出预测概率
            pred_pairs = ftmodel.predict_proba(lines)
            predict_list = [x[0] for x in pred_pairs]
        else:
            # facebook的python版本是返回两个列表
            pred_list, pred_prob_list = ftmodel.predict(lines)
            predict_list = [(pred[0], pred_prob[0]) for (pred, pred_prob) in zip(pred_list, pred_prob_list)]

    # 得到的predict_list的格式：[(预测类别, 预测概率), (预测类别, 预测概率)...]
    # 保存预测值到output文件夹
    with open(output_path, "w", encoding="utf8") as fout:
        for (pred, pred_prob) in predict_list:
            pred = re.split("__", pred)[-1]     # 去掉前缀__label__
            fout.write("%s\t%f\n" % (pred, pred_prob))
    print('预测完成，已将预测值写入输出文件：', output_path)


def model_fasttext(mode, config, input_path):
    """对fasttext模型，执行指定模式
    """
    # 如果还没有训练过词向量，那么先训练一次
    if not os.path.exists(os.path.join(config.log_root, config.model_name, "train", "word_vector.vec")):
        # 确保fasttext文件夹存在
        if not os.path.exists(os.path.join(config.log_root, config.model_name, "train")):
            os.makedirs(os.path.join(config.log_root, config.model_name, "train"))
        # 再训练词向量
        pretrained_vectors(config, input_path)
    else:
        print('已存在训练好的词向量，直接载入')

    # 执行选定模式
    if mode == "train":
        mode_train_fasttext(config, input_path)
        mode_evaluate_fasttext(config, input_path)
    elif mode == "eval":
        mode_evaluate_fasttext(config, input_path)
    elif mode == "pred":
        mode_predict_fasttext(config, input_path)
    else:
        raise TypeError("没有此模式：{}\n模式共有三种：train, eval, pred\n".format(mode))


if __name__ == "__main__":
    pass