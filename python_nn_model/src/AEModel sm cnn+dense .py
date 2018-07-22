import tensorflow as tf
import numpy as np
from libs.activations import lrelu

class AEModel(object):
    def __init__(self) :
        self.HIDDEN_STATE_SIZE = 128
        self.SCREEN_HEIGHT = 210
        self.SCREEN_WIDTH = 160
        self.n_filters = [1, 16, 32, 64]
        self.filter_sizes = 3
        self.input_shape = [None, 33600]
        self.keep_prob = 0.8

        self.build()


    def build(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, self.input_shape, name='x')
        current_input = self.x/255
        current_input = tf.reshape(current_input, [-1, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 1])

        
        encoder = []
        shapes = []
        with tf.variable_scope("encoder"):
          with tf.variable_scope("cnn_layers"):
            for layer_i, n_output in enumerate(self.n_filters[1:]):
                n_input = current_input.get_shape().as_list()[3]
                shapes.append(current_input.get_shape().as_list())
                w = tf.Variable(tf.random_normal([self.filter_sizes, self.filter_sizes, n_input, n_output], stddev=0.1))
                b = tf.Variable(tf.random_normal([n_output], stddev=0.1))
                encoder.append(w)
                conv = tf.sigmoid(tf.add(tf.nn.conv2d(current_input, w, strides=[1, 2, 2, 1], padding='SAME'), b))
                drop = tf.nn.dropout(conv, keep_prob = self.keep_prob)
                current_input = drop
            cnn_output_shape = current_input.get_shape().as_list()
            cnn_output_size = cnn_output_shape[1] * cnn_output_shape[2] * cnn_output_shape[3]
          with tf.variable_scope("dense_layers"):
            flatten = tf.contrib.layers.flatten(inputs = current_input)
            dense1 = tf.contrib.layers.fully_connected(flatten, num_outputs=256, activation_fn=tf.sigmoid)
            dense_drop1 = tf.contrib.layers.dropout(inputs = dense1, keep_prob = self.keep_prob)
                
            dense2 = tf.contrib.layers.fully_connected(inputs = dense_drop1, num_outputs=self.HIDDEN_STATE_SIZE, activation_fn=tf.sigmoid)
            dense_drop2 = tf.contrib.layers.dropout(inputs = dense2, keep_prob = self.keep_prob)
        
        with tf.variable_scope("hidden"):
          dense_drop2 = dense_drop2 * 255
          self.hidden = tf.cast(dense_drop2, tf.int32)

        encoder.reverse()
        shapes.reverse()
        with tf.variable_scope("decoder"):
          with tf.variable_scope("dense_layers"):
            dense3 = tf.contrib.layers.fully_connected(inputs = dense2, num_outputs = 256, activation_fn=tf.sigmoid)
            dense_drop3 = tf.contrib.layers.dropout(inputs = dense3, keep_prob = self.keep_prob)
                
            dense4 = tf.contrib.layers.fully_connected(inputs = dense_drop3, num_outputs = cnn_output_size, activation_fn=tf.sigmoid)
            dense_drop4 = tf.contrib.layers.dropout(inputs = dense4, keep_prob = self.keep_prob)

            reshape = tf.reshape(dense_drop4, [-1, cnn_output_shape[1], cnn_output_shape[2], cnn_output_shape[3]])
          with tf.variable_scope("cnn_transpose_layers"):
            for layer_i, shape in enumerate(shapes):
                w = encoder[layer_i]
                b = tf.Variable(tf.random_normal([w.get_shape().as_list()[2]], stddev=0.1))
                conv_transpose = tf.sigmoid(tf.add(tf.nn.conv2d_transpose(current_input, w,
                        tf.stack([tf.shape(self.x)[0], shape[1], shape[2], shape[3]]),
                        strides=[1, 2, 2, 1],
                        padding='SAME'), b))
                drop = tf.nn.dropout(conv_transpose, self.keep_prob)
                current_input = drop
        current_input = current_input * 255
        self.predict = tf.reshape(current_input, [-1, self.input_shape[1]], name = "predict")

        with tf.variable_scope("cost"):
          self.cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.predict, labels = self.x, name="cost")
          tf.summary.scalar("cost", self.cost)

        with tf.variable_scope("optimize"):
          learning_rate = 0.01
          self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)

        self.merged = tf.summary.merge_all()