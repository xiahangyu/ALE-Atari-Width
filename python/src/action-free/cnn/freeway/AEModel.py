import tensorflow as tf
import numpy as np
from layers import layer

class AEModel(object):
    def __init__(self) : #, mean_img = np.zeros([1, 33600])
        #mean_img = np.reshape(mean_img, newshape=[1, 33600])

        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, 33600], name='x')

        #self.mean = tf.Variable(mean_img, trainable=False, dtype=tf.float32)
        #self.x_mean = (self.x-self.mean)

        self.train_nn()

        self.merged = tf.summary.merge_all()


    def train_nn(self):
        #encode
        encode, conv_shapes = layer.conv_encoder(self.x)

        #hidden state
        #self.hidden = tf.reshape(encode, [-1, 53*40])
        self.hidden = tf.cast(tf.round(encode), tf.int32, name = "hidden")

        #decode
        decode = layer.conv_decoder(encode, conv_shapes)
        x_hat = decode #+self.mean
        self.x_hat = tf.cast(tf.round(x_hat), tf.int32, name="x_hat") 

        with tf.variable_scope("cost"):
            self.cost = tf.reduce_mean(tf.reduce_mean(tf.square(x_hat - self.x), 1),  name="cost") 
            tf.summary.scalar("cost",self.cost)

        with tf.variable_scope("optimize"):
            learning_rate = 0.0001
            #self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate, name="optimizer").minimize(self.cost)