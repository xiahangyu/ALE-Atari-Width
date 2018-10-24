import tensorflow as tf
import numpy as np

class layer():

    #Inputs: 
    #   screens : tensor, [None, 33600]
    #Outputs:
    #   encode : tensor, [None, 1024]
    #   cnn_shapes : python list, shapes of each convolution layers
    def conv_encoder(screens):
        screens = tf.reshape(screens, [-1, 210, 160, 1])

        cnn_shapes = []
        with tf.variable_scope("training"):
            with tf.variable_scope("cnn_layers", reuse=tf.AUTO_REUSE):
                cnn_shapes.append(screens.get_shape().as_list())
                w1 = tf.get_variable("conv1_weights", [4, 4, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
                conv1 = tf.nn.relu(tf.nn.conv2d(screens, w1, strides=[1, 2, 2, 1], padding='SAME'))
                cnn_shapes.append(conv1.get_shape().as_list())

                w2 = tf.get_variable("conv2_weights", [6, 6, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
                conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, strides=[1, 2, 2, 1], padding='SAME'))
                cnn_shapes.append(conv2.get_shape().as_list())

                w3 = tf.get_variable("conv3_weights", [6, 6, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
                conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w3, strides=[1, 2, 2, 1], padding='SAME'))
                cnn_shapes.append(conv3.get_shape().as_list())

                w4 = tf.get_variable("conv4_weights", [8, 8, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
                conv4 = tf.nn.relu(tf.nn.conv2d(conv3, w4, strides=[1, 2, 2, 1], padding='SAME'))
                cnn_shapes.append(conv4.get_shape().as_list())
            with tf.variable_scope("flatten_layers", reuse=tf.AUTO_REUSE):
                flatten = tf.contrib.layers.flatten(inputs = conv4)

                # w51 = tf.get_variable("dense1_weights", [27*20*8, 4096], initializer=tf.contrib.layers.xavier_initializer())
                # b51 = tf.get_variable("dense1_bias", [4096], initializer=tf.zeros_initializer())
                # encode1 = tf.matmul(flatten, w51) + b51

                # w52 = tf.get_variable("dense2_weights", [4096, 2048], initializer=tf.contrib.layers.xavier_initializer())
                # b52 = tf.get_variable("dense2_bias", [2048], initializer=tf.zeros_initializer())
                # encode2 = tf.nn.relu(tf.matmul(encode1, w52) + b52)
        return flatten, cnn_shapes


    #Inputs:
    #   pred_encode : tensor, [None, 1024]
    #   cnn_shapes : python list, shapes of each convolution layers
    #Outputs:
    #   decode : tensor, [None, 33600]
    def conv_decoder(pred_encode, cnn_shapes):
        with tf.variable_scope("training"):
            with tf.variable_scope("dense_layers", reuse=tf.AUTO_REUSE):
                # w01 = tf.get_variable("dense1_weights", [2048, 4096], initializer=tf.contrib.layers.xavier_initializer())
                # b01 = tf.get_variable("dense1_bias", [4096], initializer=tf.zeros_initializer())
                # dense1 = tf.matmul(pred_encode, w01) + b01

                # w02 = tf.get_variable("dense2_weights", [4096, 27*20*8], initializer=tf.contrib.layers.xavier_initializer())
                # b02 = tf.get_variable("dense2_bias", [27*20*8], initializer=tf.zeros_initializer())
                # dense2 = tf.nn.relu(tf.matmul(dense1, w02) + b02)

                conv_trans_input = tf.reshape(pred_encode, tf.stack([tf.shape(pred_encode)[0], cnn_shapes[4][1], cnn_shapes[4][2], cnn_shapes[4][3]]))

            with tf.variable_scope("cnn_layers", reuse=tf.AUTO_REUSE):
                w1 = tf.get_variable("conv4_weights", [8, 8, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
                conv_transpose1 = tf.nn.relu(tf.nn.conv2d_transpose(conv_trans_input, w1,
                                                                    tf.stack([tf.shape(pred_encode)[0], cnn_shapes[3][1], cnn_shapes[3][2], cnn_shapes[3][3]]),
                                                                    strides=[1, 2, 2, 1], padding='SAME'))

                w2 = tf.get_variable("conv_trans2_weights", [6, 6, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
                conv_transpose2 = tf.nn.relu(tf.nn.conv2d_transpose(conv_transpose1, w2,
                                                                    tf.stack([tf.shape(pred_encode)[0], cnn_shapes[2][1], cnn_shapes[2][2], cnn_shapes[2][3]]),
                                                                    strides=[1, 2, 2, 1],padding='SAME'))

                w3 = tf.get_variable("conv_trans3_weights", [6, 6, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
                conv_transpose3 = tf.nn.relu(tf.nn.conv2d_transpose(conv_transpose2, w3, 
                                                                    tf.stack([tf.shape(pred_encode)[0], cnn_shapes[1][1], cnn_shapes[1][2], cnn_shapes[1][3]]), 
                                                                    strides=[1, 2, 2, 1],padding='SAME'))

                w4 = tf.get_variable("conv_trans4_weights", [4, 4, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
                conv_transpose4 = tf.nn.relu(tf.nn.conv2d_transpose(conv_transpose3, w4, 
                                                                 tf.stack([tf.shape(pred_encode)[0], cnn_shapes[0][1], cnn_shapes[0][2], 1]), 
                                                                 strides=[1, 2, 2, 1],padding='SAME'))

        decode = tf.reshape(conv_transpose4, [-1, 33600])
        return decode