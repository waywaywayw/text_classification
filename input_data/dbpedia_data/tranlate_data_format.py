# -*- coding: utf-8 -*-
"""
将原始的PTB数据集 转化为 fasttext输入格式的文件
@author: weijiawei
@date: 2018-09-08
"""

import os
import pandas as pd

col_names = ["class", "title", "content"]


def tranlate(name):
    csv_file = pd.read_csv(os.path.join("dbpedia_csv", "%s.csv" % name), names=col_names)
    # csv_file = csv_file.sample(frac=0.1)
    # 删掉不需要的列
    csv_file = csv_file.drop(["title"], axis=1)
    # 加前缀
    csv_file["class"] = csv_file["class"].map(lambda x: "__label__" + str(x))
    # 保存
    with open(
        os.path.join("data_%s.txt" % name), "w", encoding="utf8"
    ) as fout:
        for idx, row in csv_file.iterrows():
            fout.write(str(row["class"]) + " " + str(row["content"]) + "\n")


if __name__ == "__main__":
    print('开始转换格式')
    tranlate("train")
    print('train已完成')
    tranlate("test")
    print('test已完成')
