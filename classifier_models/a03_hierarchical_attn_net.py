# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-10-09
"""
import tensorflow as tf
from classifier_models.modules.attention import attention
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from classifier_models.classifer import Classifer


class HAN(Classifer):
    def __init__(self, config):
        Classifer.__init__(self, config)

    def _build_graph(self):
        config = self.config
        # 单向rnn
        rnn_outputs, _ = tf.nn.dynamic_rnn(
            self.rnn_cell(config), self.batch_embedded, dtype=tf.float32
        )
        # Attention
        attention_output, alphas = attention(
            rnn_outputs, config.attention_size, return_alphas=True
        )
        drop = tf.nn.dropout(attention_output, self.keep_prob)
        shape = drop.get_shape()

        # Fully connected layer（dense layer)
        W = tf.Variable(tf.truncated_normal([shape[1].value, config.n_class], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[config.n_class]))
        self.y_hat = tf.nn.xw_plus_b(drop, W, b)

        # 定义loss 和 train_op
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_hat, labels=self.label)
        )
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).\
            minimize(self.loss,global_step=self.global_step)
