# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018/10/26
"""
import os
import re
import numpy as np
import pandas as pd


class EarlyStopping(object):
    def __init__(self, patience=10, mode='max'):
        self.patience = patience
        self.mode = mode
        self.pre_monitor = None
        self.cur_patience = 0  # 已经满足stopping条件的数量

    def add_monitor(self, monitor):
        """增加一个监控值。
        如果满足stop条件，那么返回True；反正返回False
        """
        if not self.pre_monitor or self._judge(monitor, self.pre_monitor):
            self.pre_monitor = monitor
            self.cur_patience = 0
            return True
        else:
            self.cur_patience += 1
            return False

    def stop(self):
        # 是否满足stop条件
        if self.cur_patience >= self.patience:
            return True
        else:
            return False

    def _judge(self, v1, v2):
        if self.mode == 'max':
            if v1 > v2:
                return True
            else:
                return False
        elif self.mode == 'min':
            if v1 < v2:
                return True
            else:
                return False


if __name__ == "__main__":
    pass
