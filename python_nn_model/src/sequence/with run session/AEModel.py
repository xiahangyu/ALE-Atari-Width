import tensorflow as tf
import numpy as np
import constants as const
from layers import layer

K = const.K
NUM_STEP = const.NUM_STEP
T = const.T
BATCH_SIZE = const.BATCH_SIZE

class AEModel(object):
    def __init__(self, mean_img = np.zeros([1, 1, 33600])) :
        mean_img = np.reshape(mean_img, newshape=[1, 1, 33600])

        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, K, 33600], name='x')
        self.y = tf.placeholder(tf.float32, [None, 1, 33600], name='y')
        self.act = tf.placeholder(tf.float32, [None, 18], name='act')

        self.mean = tf.Variable(mean_img, trainable=False, dtype=tf.float32)
        self.x_mean = (self.x-self.mean)/255

        self.train_nn()


    def train_nn(self):
        with tf.variable_scope("train_nn"):
            current_k_screens = self.x_mean
            encode, conv_shapes = layer.conv_encoder(current_k_screens)
            pred_encode = layer.action_transform(encode, self.act)
            pred = layer.conv_decoder(pred_encode, conv_shapes)
            self.pred = tf.cast(pred, tf.int32, name = "pred")

        with tf.variable_scope("cost"):
            print("y:", self.y.get_shape().as_list())
            print("pred:", pred.get_shape().as_list())
            self.cost = tf.reduce_mean(tf.square(pred-self.y), name = "cost")

        with tf.variable_scope("optimize"):
            learning_rate = 0.001
            self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)
        
