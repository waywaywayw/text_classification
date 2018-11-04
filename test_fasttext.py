# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018/10/8
"""

import os
from config.a06_fasttext import fasttext_config
from classifier_models.a06_fasttext import model_fasttext


if __name__ == "__main__":
    mode = "eval"
    config = fasttext_config()
    input_path = os.path.join("input_data", "cantonese_data")

    model_fasttext(mode, config, input_path)
