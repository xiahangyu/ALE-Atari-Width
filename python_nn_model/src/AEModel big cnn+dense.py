import tensorflow as tf
import numpy as np

class AEModel(object):
    def __init__(self):
        self.SCREEN_HEIGHT = 210
        self.SCREEN_WIDTH = 160
        self.HIDDEN_STATE_SIZE = 128
        
        self.build()
        
        
    def build(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, self.SCREEN_HEIGHT*self.SCREEN_WIDTH], name='x')
        self.keep_prob = tf.placeholder(tf.float32)
        self.input = self.x/255
        self.input = tf.reshape(self.input, [-1, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 1], name='input')

        # %%
        # Build the encoder
        with tf.variable_scope('encoder'):
            with tf.variable_scope('cnn_layers'):
                self.conv1 = tf.contrib.layers.conv2d(inputs = self.input,
                                                      num_outputs = 16,
                                                      kernel_size = 3,
                                                      stride = 2,
                                                      padding = 'SAME',
                                                      activation_fn = tf.sigmoid)
                #self.maxp1 = tf.contrib.layers.max_pool2d(inputs = self.conv1, kernel_size = 2, stride=1, padding='SAME')
                self.conv_drop1 = tf.contrib.layers.dropout(inputs = self.conv1, keep_prob = self.keep_prob)

                self.conv2 = tf.contrib.layers.conv2d(inputs = self.conv_drop1,
                                                      num_outputs = 32,
                                                      kernel_size = 3,
                                                      stride = 2,
                                                      padding = 'SAME',
                                                      activation_fn = tf.sigmoid)
                #self.maxp2 = tf.contrib.layers.max_pool2d(inputs = self.conv2, kernel_size = 2, stride=1, padding='SAME')
                self.conv_drop2 = tf.contrib.layers.dropout(inputs = self.conv2, keep_prob = self.keep_prob)

                self.conv3 = tf.contrib.layers.conv2d(inputs = self.conv_drop2,
                                                      num_outputs = 64,
                                                      kernel_size = 3,
                                                      stride = 2,
                                                      padding = 'SAME',
                                                      activation_fn = tf.sigmoid)
                #self.maxp3 = tf.contrib.layers.max_pool2d(inputs = self.conv3, kernel_size = 2, stride=1, padding='SAME')
                self.conv_drop3 = tf.contrib.layers.dropout(inputs = self.conv3, keep_prob = self.keep_prob)
                
                self.conv_drop3_shape = self.conv_drop3.get_shape().as_list()    #self.conv_drop3_shape = tf.shape(self.conv_drop3)
                self.conv_drop3_size = self.conv_drop3_shape[1] * self.conv_drop3_shape[2] * self.conv_drop3_shape[3]
            with tf.variable_scope('dense_layer'):
                self.flatten = tf.contrib.layers.flatten(inputs = self.conv_drop3)
                self.dense1 = tf.contrib.layers.fully_connected(inputs = self.flatten, num_outputs=1024, activation_fn=tf.sigmoid)
                self.dense_drop1 = tf.contrib.layers.dropout(inputs = self.dense1, keep_prob = self.keep_prob)
                
                self.dense2 = tf.contrib.layers.fully_connected(inputs = self.dense_drop1, num_outputs=512, activation_fn=tf.sigmoid)
                self.dense_drop2 = tf.contrib.layers.dropout(inputs = self.dense2, keep_prob = self.keep_prob)
                
                self.dense3 = tf.contrib.layers.fully_connected(inputs = self.dense2, num_outputs = self.HIDDEN_STATE_SIZE, activation_fn=tf.sigmoid)
                
        with tf.variable_scope('hidden_states'):
                self.dense3 = self.dense3 * 255
                self.hidden = tf.cast(self.dense3, tf.int32)
                
        with tf.variable_scope('decoder'):
            with tf.variable_scope('dense_layers'):
                self.dense4 = tf.contrib.layers.fully_connected(inputs = self.dense3, num_outputs = 512, activation_fn=tf.sigmoid)
                self.dense_drop4 = tf.contrib.layers.dropout(inputs = self.dense4, keep_prob = self.keep_prob)
                
                self.dense5 = tf.contrib.layers.fully_connected(inputs = self.dense_drop4, num_outputs = 1024, activation_fn=tf.sigmoid)
                self.dense_drop5 = tf.contrib.layers.dropout(inputs = self.dense5, keep_prob = self.keep_prob)
                
                self.dense6 = tf.contrib.layers.fully_connected(inputs = self.dense_drop5, num_outputs = self.conv_drop3_size, activation_fn=tf.sigmoid)
                self.dense_drop6 = tf.contrib.layers.dropout(inputs = self.dense6, keep_prob = self.keep_prob)
            with tf.variable_scope('cnn_transpose_layers'):
                self.reshape = tf.reshape(self.dense_drop6, [-1, self.conv_drop3_shape[1], self.conv_drop3_shape[2], self.conv_drop3_shape[3]])
    
                self.cnn_trans1 = tf.contrib.layers.conv2d_transpose(inputs = self.reshape,
                                                                     num_outputs = 32, 
                                                                     kernel_size = 3,
                                                                     stride = 2,
                                                                     padding = 'SAME',
                                                                     activation_fn = tf.sigmoid)
                self.cnn_trans_drop1 = tf.contrib.layers.dropout(inputs = self.cnn_trans1, keep_prob = self.keep_prob)
                
                self.cnn_trans2 = tf.contrib.layers.conv2d_transpose(inputs = self.cnn_trans_drop1,
                                                                     num_outputs = 16, 
                                                                     kernel_size = 3,
                                                                     stride = 2,
                                                                     padding = 'SAME',
                                                                     activation_fn = tf.sigmoid)
                self.cnn_trans_drop2 = tf.contrib.layers.dropout(inputs = self.cnn_trans2, keep_prob = self.keep_prob)
                
                self.cnn_trans3 = tf.contrib.layers.conv2d_transpose(inputs = self.cnn_trans_drop2,
                                                                     num_outputs = 1, 
                                                                     kernel_size = 3,
                                                                     stride = 2,
                                                                     padding = 'SAME',
                                                                     activation_fn = tf.sigmoid)
                self.cnn_trans_drop3 = tf.contrib.layers.dropout(inputs = self.cnn_trans3, keep_prob = self.keep_prob)
            with tf.variable_scope('predict'):
                self.flatten = tf.contrib.layers.flatten(inputs = self.cnn_trans_drop3)
                self.predict = self.flatten * 255
        
        with tf.variable_scope('loss'):
                #self.cost = tf.reduce_sum(tf.square(self.predict - self.x), name="cost")
                self.cost = sigmoid_cross_entropy_with_logits(logits = self.predict, labels = self.x, name = "cost")
                tf.summary.scalar("cost", self.cost)

        with tf.variable_scope('optimizer'):
                self.learning_rate = 0.001
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.merged = tf.summary.merge_all()