# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-18
"""
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from classifier_models.classifer import Classifer


class AttnBiLSTM(Classifer):
    def __init__(self, config):
        Classifer.__init__(self, config)

    def _build_graph(self):
        config = self.config
        # 定义双向rnn
        rnn_outputs, _ = bi_rnn(
            self.rnn_cell(config),
            self.rnn_cell(config),
            inputs = self.batch_embedded,
            dtype = tf.float32,
        )

        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([config.hidden_size], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        alpha = tf.nn.softmax(
            tf.matmul(tf.reshape(M, [-1, config.hidden_size]), tf.reshape(W, [-1, 1]))
        )
        r = tf.matmul(
            tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, config.max_len, 1])
        )
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, self.keep_prob)

        # Fully connected layer(dense layer)
        FC_W = tf.Variable(
            tf.truncated_normal([config.hidden_size, config.n_class], stddev=0.1)
        )
        FC_b = tf.Variable(
            tf.constant(0., shape=[config.n_class])
        )
        self.y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)

        # 定义loss 和 train_op
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.y_hat, labels=self.label
            )
        )

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(
            loss_to_minimize,
            tvars,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
        )
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name="train_step"
        )



