# -*- coding: utf-8 -*-

from utils.zhtools.langconv import Converter
import zhconv

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


if __name__ == "__main__":
    traditional_sentence = '憂郁的臺灣烏龜'
    simplified_sentence = Traditional2Simplified(traditional_sentence)
    print(simplified_sentence)

    # traditional_sentence = '基地组织会显著提高'
    print(zhconv.convert(traditional_sentence, 'zh-cn'))
    '''
    输出结果：
        忧郁的台湾乌龟
    '''
