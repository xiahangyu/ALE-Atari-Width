import tensorflow as tf
import numpy as np
from layers import layer


class AEModel(object):
    def __init__(self) : #, mean_img = np.zeros([1, 33600])
        #mean_img = np.reshape(mean_img, newshape=[1, 33600])

        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, 33600], name='x')
        self.y = tf.placeholder(tf.float32, [None, 33600], name='y')
        self.act = tf.placeholder(tf.float32, [None,18], name='act')

        self.train_nn()

        self.merged = tf.summary.merge_all()


    def train_nn(self):
        #encode
        encode, conv_shapes = layer.conv_encoder(self.x)

        #action_transform
        pred_encode, encoder_factor = layer.action_transform(encode, self.act)
        self.hidden2 = tf.cast(tf.round(encoder_factor), tf.int32, name="hidden1")

        #decode
        decode = layer.conv_decoder(pred_encode, conv_shapes)
        self.y_hat = tf.identity(decode, name="y_hat") 

        with tf.variable_scope("cost"):
            self.cost = tf.reduce_mean( tf.reduce_mean( tf.square(self.y_hat - self.y), 1), name="cost")
            tf.summary.scalar('cost', self.cost)

        with tf.variable_scope("optimize"):
            learning_rate = 0.001
            self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)
