# -*- coding: utf-8 -*-
"""
@author: weijiawei
@date: 2018-09-21
"""

from abc import ABCMeta, abstractmethod
import tensorflow as tf


class Classifer(object):
    """
    提供分类模型一些公用的模块。比如：
    RNNCell的选择；
    验证实现的模型是否存在 必须有的接口op, 比如self.x, self.label, self.loss .. 等等
    embedding层、placeholder层、summary层
    还有啥？
    """
    __metaclass__ = ABCMeta     # 虚函数声明

    def __init__(self, config):
        self.config = config
        # 定义必需的op
        # placeholder_layer
        self.x = None   # 输入数据
        self.label = None   # 输入数据的标签
        self.keep_prob = None   # dropout
        # embedding_layer
        self.batch_embedded = None  # 经过了embed层的input
        # build_graph
        self.global_step = None # 当前训练step数
        self.y_hat = None
        self.train_op = None
        self.loss = None
        self.predict_prob = None
        self.predict = None
        self.real_label = None  # 真实的标签。如果self.label是one-hot表示，那么转换回 类别表示
        self.accuracy = None
        # summary_layer
        self.summaries = None

    def build_graph(self):
        """
        构建图结构。
        大致包括placeholder_layer, embedding_layer, summary_layer等
        """
        print("开始构建tf图")
        self.placeholder_layer()
        self.embedding_layer()
        self.global_step = tf.train.get_or_create_global_step()
        self._build_graph()  # 构建关键的中间部分
        self.acc_metric_layer()
        self.summary_layer()
        self.test_graph_interface() # 测试是否需要的op都有了
        print("tf图 构建成功!")

    @abstractmethod
    def _build_graph(self):
        """每个分类模型的核心部分"""
        pass

    def placeholder_layer(self):
        """定义所需的placeholder张量
        """
        config = self.config
        # placeholder
        self.x = tf.placeholder(tf.int32, [None, config.max_len])
        # 根据label是否为one-hot表示，定义不同的placeholder
        if config.one_hot:
            self.label = tf.placeholder(tf.float32, [None, config.n_class])
        else:
            self.label = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)

    def embedding_layer(self):
        """定义embedding_layer的张量
        """
        config = self.config
        # Word embedding
        embeddings_var = tf.Variable(
            tf.random_uniform([config.vocab_size, config.embedding_size], -1.0, 1.0),
            trainable=True,
        )
        self.batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
        # print(batch_embedded.shape)  # (?, 256, 100)

    def acc_metric_layer(self):
        """定义预测值、预测概率、和精确率(acc)的张量
        """
        # Accuracy metric
        self.predict_prob = tf.nn.softmax(self.y_hat)
        self.predict = tf.argmax(tf.nn.softmax(self.y_hat), 1, output_type=tf.int64) # int64?
        # 根据label是否为one-hot表示，统计label的方式不同
        if self.config.one_hot:
            self.real_label = tf.argmax(self.label, 1)
        else:
            self.real_label = self.label
        correct_prediction = tf.equal(self.predict, self.real_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def summary_layer(self):
        """定义需要显示的summary张量，和总的merge_all张量
        """
        # loss, acc
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        # lr?
        # global_norm?
        self.summaries = tf.summary.merge_all()

    def test_graph_interface(self):
        """
        用于检验构建的图是否含有必需的op. (比如必须有self.train_op, self.loss)
        图构建完毕后，调用该方法来检验"""
        # placeholder_layer
        assert self.x is not None
        assert self.label is not None
        assert self.keep_prob is not None
        # embedding_layer
        assert self.batch_embedded is not None
        # build_graph_layer
        assert self.global_step is not None
        assert self.train_op is not None
        assert self.loss is not None
        assert self.predict_prob is not None
        assert self.predict is not None
        assert self.accuracy is not None
        # summary_layer
        # 怎么把summary记录的op也写进来？
        assert self.summaries is not None

    def rnn_cell(self, config):
        """根据配置文件，选择指定的RNNcell类型
        """
        if config.RNNCell == "BASIC":
            return tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size)
        elif config.RNNCell == "BLOCK":
            return tf.contrib.rnn.LSTMBlockCell(config.hidden_size)
        elif config.RNNCell == "GRU":
            return tf.nn.rnn_cell.GRUCell(config.hidden_size)
        else:
            raise ValueError("rnn_mode %s not supported" % config.RNNCell)
