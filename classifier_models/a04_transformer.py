# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-10-11
"""
import tensorflow as tf
from classifier_models.modules.multihead import feedforward, multihead_attention

from classifier_models.classifer import Classifer


class Transformer(Classifer):
    def __init__(self, config):
        Classifer.__init__(self, config)

    def _build_graph(self):
        config = self.config
        # multi-head attention      self-attention?
        ma = multihead_attention(queries=self.batch_embedded, keys=self.batch_embedded)
        # FFN(x) = LN(x + point-wisely NN(x))
        outputs = feedforward(ma, [config.hidden_size, config.embedding_size])
        outputs = tf.reshape(outputs, [-1, config.max_len * config.embedding_size])
        self.y_hat = tf.layers.dense(outputs, units=config.n_class)

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



