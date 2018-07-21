# import tensorflow as tf
# import math
# import numpy as np
# from libs.activations import lrelu

# screen_height = 210
# screen_width = 160


# class AEModel(object):
#     def __init__(self,
#                  input_shape=[None, 33600],
#                  n_filters=[1, 16, 32, 64],
#                  filter_sizes=[3, 3, 3, 3],
#                  mean_img=np.zeros([1, 33600])) :
#         tf.reset_default_graph()

#         self.x = tf.placeholder(tf.float32, input_shape, name='x')
#         self.mean = tf.Variable(mean_img, trainable=False, dtype=tf.float32)
#         x_tensor = tf.subtract(self.x, self.mean)

#         # %%
#         # ensure 2-d is converted to square tensor.
#         if len(x_tensor.get_shape()) == 2:
#             x_tensor = tf.reshape(x_tensor, [-1, screen_height, screen_width, n_filters[0]])
#         else:
#             raise ValueError('Unsupported input dimensions')
#         current_input = x_tensor

#         # %%
#         # Build the encoder
#         encoder = []
#         shapes = []
#         for layer_i, n_output in enumerate(n_filters[1:]):
#             n_input = current_input.get_shape().as_list()[3]
#             shapes.append(current_input.get_shape().as_list())
#             w = tf.Variable(
#                 tf.random_uniform([
#                     filter_sizes[layer_i],
#                     filter_sizes[layer_i],
#                     n_input, n_output],
#                     -1.0 / math.sqrt(n_input),
#                     1.0 / math.sqrt(n_input)))
#             b = tf.Variable(tf.zeros([n_output]))
#             encoder.append(w)
#             output = lrelu(tf.add(tf.nn.conv2d(current_input, w, strides=[1, 2, 2, 1], padding='SAME'), b))
#             current_input = output
#         # %%
#         # store the latent representation
#         self.z = tf.sigmoid(current_input)
#         self.z = tf.multiply(self.z, 255)
#         self.z = tf.cast(tf.reshape(self.z, [-1, 34560]), tf.int32)
#         tf.summary.histogram("Hidden_hist", self.z)
#         encoder.reverse()
#         shapes.reverse()

#         # %%
#         # Build the decoder using the same weights
#         for layer_i, shape in enumerate(shapes):
#             w = encoder[layer_i]
#             b = tf.Variable(tf.zeros([w.get_shape().as_list()[2]]))
#             output = tf.sigmoid(tf.add(
#                 tf.nn.conv2d_transpose(
#                     current_input, w,
#                     tf.stack([tf.shape(x_tensor)[0], shape[1], shape[2], shape[3]]),
#                     strides=[1, 2, 2, 1], padding='SAME'), b))
#             current_input = output

#         # %%
#         # now have the reconstruction through the network
#         y_tensor = current_input
#         self.y = tf.cast(tf.add(tf.reshape(y_tensor, [-1, 33600]), self.mean, name="y"), tf.int32)
#         # cost function measures pixel-wise difference
#         self.cost = tf.reduce_sum(tf.square(y_tensor - x_tensor), name="cost")
#         tf.summary.scalar("Cost_scalar", self.cost)
#         tf.summary.histogram("Cost_hist", self.cost)

#         # %%
#         learning_rate = 0.01
#         self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)

#         self.merged = tf.summary.merge_all()


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
        self.y = tf.placeholder(tf.float32, [None, self.SCREEN_HEIGHT*self.SCREEN_WIDTH], name ='y')
        self.keep_prob = tf.placeholder(tf.float32)
        self.input = self.x/255
        self.input = tf.reshape(self.input, [-1, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 1], name='input')

        # %%
        # Build the encoder
        with tf.variable_scope('encoder'):
            with tf.variable_scope('cnn_layers'):
                self.conv1 = tf.contrib.layers.conv2d(inputs = self.input,
                                                      num_outputs = 32,
                                                      kernel_size = 3,
                                                      stride = 1,
                                                      padding = 'SAME',
                                                      activation_fn = tf.sigmoid)
                #self.maxp1 = tf.contrib.layers.max_pool2d(inputs = self.conv1, kernel_size = 2, stride=1, padding='SAME')
                self.conv_drop1 = tf.contrib.layers.dropout(inputs = self.conv1, keep_prob = self.keep_prob)

                self.conv2 = tf.contrib.layers.conv2d(inputs = self.conv_drop1,
                                                      num_outputs = 64,
                                                      kernel_size = 3,
                                                      stride = 1,
                                                      padding = 'SAME',
                                                      activation_fn = tf.sigmoid)
                #self.maxp2 = tf.contrib.layers.max_pool2d(inputs = self.conv2, kernel_size = 2, stride=1, padding='SAME')
                self.conv_drop2 = tf.contrib.layers.dropout(inputs = self.conv2, keep_prob = self.keep_prob)

                self.conv3 = tf.contrib.layers.conv2d(inputs = self.conv_drop2,
                                                      num_outputs = 128,
                                                      kernel_size = 3,
                                                      stride = 1,
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
                self.hidden = self.dense3 * 255
                self.hidden = tf.cast(self.hidden, tf.int32)
                
        with tf.variable_scope('decoder'):
            with tf.variable_scope('dense_layers'):
                self.dense4 = tf.contrib.layers.fully_connected(inputs = self.hidden, num_outputs = 512, activation_fn=tf.nn.sigmoid)
                self.dense_drop4 = tf.contrib.layers.dropout(inputs = self.dense4, keep_prob = self.keep_prob)
                
                self.dense5 = tf.contrib.layers.fully_connected(inputs = self.dense_drop4, num_outputs = 1024, activation_fn=tf.nn.sigmoid)
                self.dense_drop5 = tf.contrib.layers.dropout(inputs = self.dense5, keep_prob = self.keep_prob)
                
                self.dense6 = tf.contrib.layers.fully_connected(inputs = self.dense_drop5, num_outputs = self.conv_drop3_size, activation_fn=tf.nn.sigmoid)
                self.dense_drop6 = tf.contrib.layers.dropout(inputs = self.dense6, keep_prob = self.keep_prob)
            with tf.variable_scope('cnn_transpose_layers'):
                self.reshape = tf.reshape(self.dense_drop6, [-1, self.conv_drop3_shape[1], self.conv_drop3_shape[2], self.conv_drop3_shape[3]])
    
                self.cnn_trans1 = tf.contrib.layers.conv2d_transpose(inputs = self.reshape,
                                                                     num_outputs = 64, 
                                                                     kernel_size = 3,
                                                                     stride = 1,
                                                                     padding = 'SAME',
                                                                     activation_fn = tf.sigmoid)
                self.cnn_trans_drop1 = tf.contrib.layers.dropout(inputs = self.cnn_trans1, keep_prob = self.keep_prob)
                
                self.cnn_trans2 = tf.contrib.layers.conv2d_transpose(inputs = self.cnn_trans_drop1,
                                                                     num_outputs = 32, 
                                                                     kernel_size = 3,
                                                                     stride = 1,
                                                                     padding = 'SAME',
                                                                     activation_fn = tf.sigmoid)
                self.cnn_trans_drop2 = tf.contrib.layers.dropout(inputs = self.cnn_trans2, keep_prob = self.keep_prob)
                
                self.cnn_trans3 = tf.contrib.layers.conv2d_transpose(inputs = self.cnn_trans_drop2,
                                                                     num_outputs = 1, 
                                                                     kernel_size = 3,
                                                                     stride = 1,
                                                                     padding = 'SAME',
                                                                     activation_fn = tf.sigmoid)
                self.cnn_trans_drop3 = tf.contrib.layers.dropout(inputs = self.cnn_trans3, keep_prob = self.keep_prob)
            with tf.variable_scope('predict'):
                self.flatten = tf.contrib.layers.flatten(inputs = self.cnn_trans_drop3)
                self.predict = self.flatten * 255
        
        with tf.variable_scope('loss'):
                self.cost = tf.reduce_sum(tf.square(self.predict - self.y), name="cost")
                tf.summary.scalar("cost", self.cost)

        with tf.variable_scope('optimizer'):
                self.learning_rate = 0.001
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.merged = tf.summary.merge_all()

