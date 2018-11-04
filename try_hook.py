# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018/10/26
"""
import os
import re
import numpy as np
import pandas as pd

class TrainHook(tf.train.SessionRunHook):
    """Logs loss, acc and runtime."""
    def __init__(self, model, log_step, summary_writer_train):
        self._model = model
        self._log_step = log_step
        self._summary_writer_train = summary_writer_train
        self._loss_list = []
        self._acc_list = []

    def begin(self):
        self._start_time = time.time()

    def before_run(self, run_context):
        model = self._model
        to_return = {
            # 'train_op': model.train_op,
            'summaries': model.summaries,
            'acc': model.accuracy,
            'loss': model.loss,
            'global_step': model.global_step,
        }
        return tf.train.SessionRunArgs(to_return)

    def after_run(self, run_context, run_values):
        return_dict = run_values.results
        global_step = return_dict['global_step']
        # 每次 train run 完都记录acc, loss
        self._acc_list.append(return_dict['acc'])
        self._loss_list.append(return_dict['loss'])
        self._summary_writer_train.add_summary(return_dict['summaries'], global_step)
        # step达到记录点： 跑一遍验证集，得到acc ,loss
        if global_step % self._log_step == 0:
            current_time = time.time()
            duration = current_time - self._start_time  # duration持续的时间
            self._start_time = current_time
            print('\n%s: ' % datetime.now())
            print("Global_step: %d, cost time: %.3f s" % (global_step, duration))

            # 统计acc, loss
            acc_value = np.asarray(self._acc_list).mean()
            loss_value = np.asarray(self._loss_list).mean()  # 统计均值
            print("Train accuracy: %.3f %%, loss: %.4f" % (acc_value * 100, loss_value))

if __name__ == "__main__":
    pass
