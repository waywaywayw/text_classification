# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018/10/9
"""

import os
import re
import numpy as np
import pandas as pd
import pytest

from config.a01_attn_bi_lstm import AttnBiLSTM_config
from classifier_models.a01_attn_bi_lstm import AttnBiLSTM

def test_case_111():
    # 创建分类器
    config = AttnBiLSTM_config
    classifier = AttnBiLSTM(config)
    classifier.test_model_interface()
    # classifier.build_graph()
    assert hasattr(classifier, 'x')


if __name__ == "__main__":
    pytest.main()
    # test_case()