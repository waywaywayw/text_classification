# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018/10/11
"""
import os
from classifier_models.a01_attn_bi_lstm import AttnBiLSTM
from classifier_models.a02_ind_rnn import IndRNN
from classifier_models.a03_hierarchical_attn_net import HAN
from classifier_models.a04_transformer import Transformer
from classifier_models.a05_adversarial import Adversarial

from config.a01_attn_bi_lstm import AttnBiLSTM_config
from config.a02_ind_rnn import IndRNN_config
from config.a03_hierarchical_attn_net import HAN_config
from config.a04_transformer import transformer_config
from config.a05_adversarial import Adversarial_config
from config.a06_fasttext import fasttext_config


# 当前可用模型的名字
current_model_name = [
        None,     # 0索引位置 设置为空
        "attn_bi_lstm",
        "ind_rnn",
        "hierarchical_attn_lstm",
        'transformer',
        'adversarial',
        'fasttext'
    ]

# 当前可用模型
current_model = {
    current_model_name[1]: AttnBiLSTM,
    current_model_name[2]: IndRNN,
    current_model_name[3]: HAN,
    current_model_name[4]: Transformer,
    current_model_name[5]: Adversarial,
    current_model_name[6]: None     # fasttext调用的api, 故没有模型本体
}

# 当前可用模型的配置文件
current_model_config = {
    current_model_name[1]: AttnBiLSTM_config,
    current_model_name[2]: IndRNN_config,
    current_model_name[3]: HAN_config,
    current_model_name[4]: transformer_config,
    current_model_name[5]: Adversarial_config,
    current_model_name[6]: fasttext_config
}

if __name__ == "__main__":
    pass
