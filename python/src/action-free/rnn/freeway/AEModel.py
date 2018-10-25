import tensorflow as tf
import numpy as np
import constants as const
from layers import layer

BATCH_SIZE = const.BATCH_SIZE

class AEModel(object):


    def __init__(self) :
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, 33600], name='x')
        self.train_nn()

        self.merged = tf.summary.merge_all()


    def train_nn(self):
        #encode
        encode, states = layer.rnn_encoder(self.x, 160)

        self.hidden = tf.cast(tf.round(encode), tf.int32, name="hidden1")

        #action_transform
        decode = layer.rnn_decoder(encode, states, 160)  #, encode_factor
        self.x_hat = tf.identity(decode, name="x_hat")

        with tf.variable_scope("cost"):
            self.cost = tf.reduce_mean(tf.reduce_mean( tf.square(self.x-self.x_hat), 1), name="cost")
            tf.summary.scalar('cost', self.cost)

        with tf.variable_scope("optimize"):
            learning_rate = 0.0001
            self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)
