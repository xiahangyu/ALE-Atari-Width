import tensorflow as tf
import numpy as np
from libs.activations import lrelu

class AEModel(object):
    def __init__(self, mean_img = np.zeros([33600])) :
        self.HIDDEN_STATE_SIZE = 2048
        self.SCREEN_HEIGHT = 210
        self.SCREEN_WIDTH = 160
        self.input_shape = [None, 33600]
        self.keep_prob = 0.8

        mean_img = np.reshape(mean_img, newshape = [33600])
        self.build(mean_img)


    def build(self, mean_img = np.zeros([33600])):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, self.input_shape, name='x')
        self.mean = tf.Variable(mean_img, trainable=False, dtype=tf.float32, name='mean')
        x_mean = (self.x-self.mean)/255

        shapes = []
        with tf.variable_scope("encoder"):
          with tf.variable_scope("cnn_layers"):
            conv_input = tf.reshape(x_mean, [-1, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 1])

            shapes.append(conv_input.get_shape().as_list())
            w = tf.Variable(tf.random_normal([8, 8, 1, 64], stddev=0.1))
            conv1 = tf.nn.relu(tf.nn.conv2d(conv_input, w, strides=[1, 2, 2, 1], padding='SAME'))

            shapes.append(conv1.get_shape().as_list())
            w = tf.Variable(tf.random_normal([6, 6, 64, 128], stddev=0.1))
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w, strides=[1, 2, 2, 1], padding='SAME'))

            shapes.append(conv2.get_shape().as_list())
            w = tf.Variable(tf.random_normal([6, 6, 128, 128], stddev=0.1))
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w, strides=[1, 2, 2, 1], padding='SAME'))

            shapes.append(conv3.get_shape().as_list())
            w = tf.Variable(tf.random_normal([4, 4, 128, 128], stddev=0.1))
            conv4 = tf.nn.relu(tf.nn.conv2d(conv3, w, strides=[1, 2, 2, 1], padding='SAME'))

            shapes.append(conv4.get_shape().as_list())
            conv4_shape= conv4.get_shape().as_list()
          with tf.variable_scope("dense_layers"):
            flatten = tf.contrib.layers.flatten(inputs = conv4)
            flatten_size = flatten.get_shape().as_list()[1]
            dense1 = tf.contrib.layers.fully_connected(flatten, num_outputs=self.HIDDEN_STATE_SIZE, activation_fn=tf.nn.relu)
        
        with tf.variable_scope("hidden"):
            self.hidden = tf.cast(dense1, tf.int32, name = "hidden")

        with tf.variable_scope("decoder"):
          with tf.variable_scope("dense_layers"):
            dense2 = tf.contrib.layers.fully_connected(inputs = dense1, num_outputs = flatten_size, activation_fn=tf.nn.relu)
            reshape = tf.reshape(dense2, tf.stack([tf.shape(self.x)[0], conv4_shape[1], conv4_shape[2], conv4_shape[3]]))
            conv_trans_input = reshape
          with tf.variable_scope("cnn_transpose_layers"):
            w = tf.Variable(tf.random_normal([4, 4, 128, 128], stddev=0.1))
            conv_transpose1 = tf.nn.relu(tf.nn.conv2d_transpose(conv_trans_input, w,
                                                                tf.stack([tf.shape(self.x)[0], shapes[3][1], shapes[3][2], shapes[3][3]]),
                                                                strides=[1, 2, 2, 1],padding='SAME'))

            w = tf.Variable(tf.random_normal([6, 6, 128, 128], stddev=0.1))
            conv_transpose2 = tf.nn.relu(tf.nn.conv2d_transpose(conv_transpose1, w,
                                                                tf.stack([tf.shape(self.x)[0], shapes[2][1], shapes[2][2], shapes[2][3]]),
                                                                strides=[1, 2, 2, 1],padding='SAME'))

            w = tf.Variable(tf.random_normal([6, 6, 64, 128], stddev=0.1))
            conv_transpose3 = tf.nn.relu(tf.nn.conv2d_transpose(conv_transpose2, w, 
                                                                tf.stack([tf.shape(self.x)[0], shapes[1][1], shapes[1][2], shapes[1][3]]), 
                                                                strides=[1, 2, 2, 1],padding='SAME'))


            w = tf.Variable(tf.random_normal([4, 4, 1, 64], stddev=0.1))
            conv_transpose4 = tf.nn.relu(tf.nn.conv2d_transpose(conv_transpose3, w, 
                                                                tf.stack([tf.shape(self.x)[0], shapes[0][1], shapes[0][2], shapes[0][3]]), 
                                                                strides=[1, 2, 2, 1],padding='SAME'))
            self.predict = tf.reshape(conv_transpose4, [-1, self.input_shape[1]])
        with tf.variable_scope("cost"):
          self.cost = tf.reduce_sum(tf.square(self.predict - self.x), name="cost")
          tf.summary.scalar("cost", self.cost)

        with tf.variable_scope("optimize"):
          learning_rate = 0.01
          self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)

        self.merged = tf.summary.merge_all()