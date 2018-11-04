# -*- coding: utf-8 -*-
"""
将粤语句子数据 转换为 fasttext输入格式的文件
粤语句子的标签是__label__1 , 普通话的标签是__label__0
@author: weijiawei
@date: 2018-09-08
"""
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.zhtools.zhblog import zhblog
import zhconv


def convert_CN(sentence):
    """调用zhconv包，实现繁体转简体（感觉这个好一些）
    """
    if isinstance(sentence, str):
        sentence = zhconv.convert(sentence, 'zh-cn')
    return sentence


def convert_CN_v2(sentence):
    """调用整理的繁简表，实现繁体转简体
    """
    if isinstance(sentence, str):
        for idx,src in enumerate(sentence):
            target = zhblog.get(src, None)
            if target is not None:
                # print(sentence)
                sentence = sentence.replace(src, target)
                # print(sentence)
    return sentence


def process_content(data):
    """对content列进行处理
    """
    # 清理content列（去掉空白字符）
    data = re.sub('\s', '', str(data))
    # 粤语中的繁体转换成简体（2018.10.12新增）
    data = convert_CN(data)
    # char级切分词
    data = ' '.join(data)  # 用空格分割字符串
    return data


def write_to_file(file_name, data):
    with open(os.path.join(file_name), "w", encoding="utf8") as fout:
        for idx, (_, row) in enumerate(data.iterrows()):
            fout.write(row["class"] + " " + row["content"] + "\n")
            if idx % 10000 == 0:
                print('写入了{}条数据..'.format(idx))


if __name__ == "__main__":
    # 初始化DataFrame
    input_path = os.path.join("粤语句子数据.xlsx")
    data_pd = pd.read_excel(input_path)
    # data_pd = data_pd[:100]
    # data_pd = data_pd.sample(frac=0.1)

    # 开始处理DataFrame
    # 1. 构造完整的DataFrame
    data_pd_len = len(data_pd)  # 先取出原始的数据数量
    # 把普通话和粤语 合并成content列
    temp_series = pd.concat((data_pd['普通话'], data_pd['粤语']), axis=0, ignore_index=True)
    data_pd = pd.concat((data_pd, temp_series), axis=1)
    data_pd.rename(columns={0: 'content'}, inplace=True)
    # print(data_pd.columns)
    # 丢弃不需要的列
    data_pd.drop(['普通话'], axis=1, inplace=True)
    data_pd.drop(['粤语'], axis=1, inplace=True)
    data_pd.drop(['繁体'], axis=1, inplace=True)
    # 加标签
    data_pd.insert(0, 'class', "")  # 新插入一列
    data_pd['class'][:data_pd_len] = '__label__0'
    data_pd['class'][data_pd_len:] = '__label__1'

    # 2. 针对某些项逐个处理
    data_pd = data_pd.dropna(axis=0)  # 清理含有nan的行
    # 清理content项是Nan的数据（丢弃这种实现）
    # data_pd = drop_rows(data_pd, lambda x: str(x['content']) == 'nan')
    # 对content项进行处理
    data_pd['content'] = data_pd['content'].map(lambda x: process_content(x))
    # print(data_pd)

    # 处理完毕，保存DataFrame
    # 划分训练集和测试集，并保存
    train_data_pd, test_data_pd = train_test_split(data_pd, test_size=0.1, random_state=2018)
    # write train set
    write_to_file("data_train.txt", train_data_pd)
    # write test set
    write_to_file("data_test.txt", test_data_pd)