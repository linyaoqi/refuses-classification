import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')

import tensorflow
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class TextCNN(object):
    """一个文本分类的CNN。首先使用嵌入层，接着是卷积层、最大池化层和softmax层。"""
    def __init__(
      self,w2v_model, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        ##输入、输出和退出的占位符
        self.input_x = tensorflow.compat.v1.placeholder(tensorflow.compat.v1.int32, [None, sequence_length], name="input_x")
        self.input_y = tensorflow.compat.v1.placeholder(tensorflow.compat.v1.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tensorflow.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        ##嵌入层
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if w2v_model is None:
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="word_embeddings")
            else:
                self.W = tensorflow.compat.v1.get_variable("word_embeddings",
                    initializer=w2v_model.vectors.astype(np.float32))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        ##为每个过滤器尺寸创建一个卷积层和最大池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                ##卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tensorflow.compat.v1.truncated_normal(filter_shape, stddev=0.1),dtype=tf.float32, name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]),dtype=tf.float32, name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                ##应用非线性
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                ##output上的最大池化
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        ##合并所有池化特征
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        with tf.name_scope("dropout"):  ##添加退出
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        ##非标准所有的分数和预测
        with tf.name_scope("output"):
           W = tensorflow.compat.v1.get_variable(
                        "W",
                        shape=[num_filters_total, num_classes],
                        #initializer=tf.contrib.layers.xavier_initializer())
                        initializer=tf.keras.initializers.he_normal())  ##权值初始化
           b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
           l2_loss += tf.nn.l2_loss(b)         
           self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
           self.predictions = tf.argmax(self.scores, 1, name="predictions")
        ##计算交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        ##精确度
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
