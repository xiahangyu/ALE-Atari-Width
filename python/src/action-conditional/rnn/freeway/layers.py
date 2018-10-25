import tensorflow as tf
import numpy as np
import constants as const

K = const.K
N_HIDDEN = const.N_HIDDEN

class layer():

    #Inputs: 
    #   k_screens : tensor, [None, K, 33600]
    #Outputs:
    #   cnn_outputs : tensor, [None, K, 1024]
    #   cnn_shapes : python list, shapes of each convolution layers
    def conv_encoder(k_screens):
        k_screens = tf.reshape(k_screens, [-1, 1, 210, 160])
        NHWC_k_screens = tf.transpose(k_screens, [0, 2, 3, 1])

        cnn_shapes = []
        with tf.variable_scope("cnn_layers", reuse=tf.AUTO_REUSE):
            cnn_shapes.append(NHWC_k_screens.get_shape().as_list())
            w1 = tf.get_variable("conv1_weights", [4, 4, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
            conv1 = tf.nn.relu(tf.nn.conv2d(NHWC_k_screens, w1, strides=[1, 2, 2, 1], padding='SAME'))
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

            w = tf.get_variable("dense_weights", [14*10*128, N_HIDDEN], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("dense_bias", [N_HIDDEN], initializer=tf.contrib.layers.xavier_initializer())
            dense1 = tf.nn.relu(tf.matmul(flatten, w) + b)

            ww = tf.get_variable("dense1_weights", [N_HIDDEN, N_HIDDEN], initializer=tf.contrib.layers.xavier_initializer())
            bb = tf.get_variable("dense1_bias", [N_HIDDEN], initializer=tf.contrib.layers.xavier_initializer())
            dense2 = tf.nn.relu(tf.matmul(dense1, ww) + bb)

            cnn_outputs = tf.reshape(dense2, [-1, K, N_HIDDEN])
        return cnn_outputs, cnn_shapes

    #Inputs:
    #   cnn_outputs: tensor, [batch_size, K_steps, n_input]
    #Outputs:
    #   encode: tensor, [batch_size, N_HIDDEN]
    #   states: tensor, [N_HIDDEN]
    def add_rnn(cnn_outputs):
        with tf.variable_scope("rnn_layers", reuse=tf.AUTO_REUSE):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=N_HIDDEN, reuse=tf.AUTO_REUSE, name="the_only_lstm_cell")
            hiddens, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=cnn_outputs, dtype=tf.float32)
            hiddens = tf.transpose(hiddens,[1,0,2])    #tensor, [n_steps, batch_size, n_input]
            encode = tf.reshape(hiddens[-1], [-1, N_HIDDEN])
        return encode

    #Inputs:
    #   encode : tensor, [None, N_HIDDEN]
    #   act : tensor, [None, 18]
    #Outputs:
    #   pred_encoder : tensor, [None, N_HIDDEN]
    def action_transform(encode, act):
        with tf.variable_scope("action_transform", reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable("encoder_factor_weights", [N_HIDDEN, N_HIDDEN], initializer=tf.random_uniform_initializer(-1,1))
            b1 = tf.get_variable("encoder_factor_bias", [N_HIDDEN], initializer=tf.zeros_initializer())
            encode_factor = tf.matmul(encode, w1) + b1

            w2 = tf.get_variable("act_emb_weights1", [18, N_HIDDEN], initializer=tf.random_uniform_initializer(-0.1,0.1)) #tf.zeros_initializer()
            b2 = tf.get_variable("act_emb_bias1", [N_HIDDEN], initializer=tf.zeros_initializer())
            act_emb = tf.matmul(act, w2) + b2

            decode_factor = tf.multiply(encode_factor, act_emb)

            w3 = tf.get_variable("pred_encode_weights", [N_HIDDEN, N_HIDDEN], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("pred_encode_bias", [N_HIDDEN], initializer=tf.zeros_initializer())
            pred_encode = tf.matmul(decode_factor, w3) + b3

        return pred_encode, encode_factor


    #Inputs:
    #   pred_encode : tensor, [None, N_HIDDEN]
    #   cnn_shapes : python list, shapes of each convolution layers
    #Outputs:
    #   decode : tensor, [None, 1, 33600]
    def conv_decoder(pred_encode, cnn_shapes):
        with tf.variable_scope("dense_layers", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("dense_weights", [N_HIDDEN, 14*10*128], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("dense_bias", [14*10*128], initializer=tf.zeros_initializer())
            dense = tf.nn.relu(tf.matmul(pred_encode, w) + b)

            conv_trans_input = tf.reshape(dense, tf.stack([tf.shape(pred_encode)[0], cnn_shapes[4][1], cnn_shapes[4][2], cnn_shapes[4][3]]))

        with tf.variable_scope("cnn_layers", reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable("conv_transpose1_weights", [8, 8, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            conv_transpose1 = tf.nn.relu(tf.nn.conv2d_transpose(conv_trans_input, w1,
                                                                tf.stack([tf.shape(pred_encode)[0], cnn_shapes[3][1], cnn_shapes[3][2], cnn_shapes[3][3]]),
                                                                strides=[1, 2, 2, 1], padding='SAME'))

            w2 = tf.get_variable("conv_transpose2_weights", [6, 6, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            conv_transpose2 = tf.nn.relu(tf.nn.conv2d_transpose(conv_transpose1, w2,
                                                                tf.stack([tf.shape(pred_encode)[0], cnn_shapes[2][1], cnn_shapes[2][2], cnn_shapes[2][3]]),
                                                                strides=[1, 2, 2, 1],padding='SAME'))

            w3 = tf.get_variable("conv_transpose3_weights", [6, 6, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
            conv_transpose3 = tf.nn.relu(tf.nn.conv2d_transpose(conv_transpose2, w3, 
                                                                tf.stack([tf.shape(pred_encode)[0], cnn_shapes[1][1], cnn_shapes[1][2], cnn_shapes[1][3]]), 
                                                                strides=[1, 2, 2, 1],padding='SAME'))

            w4 = tf.get_variable("conv_transpose4_weights", [4, 4, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
            conv_transpose4 = tf.nn.relu(tf.nn.conv2d_transpose(conv_transpose3, w4, 
                                                             tf.stack([tf.shape(pred_encode)[0], cnn_shapes[0][1], cnn_shapes[0][2], 1]), 
                                                             strides=[1, 2, 2, 1],padding='SAME'))
            NCHW_conv_transpose4 = tf.transpose(conv_transpose4, [0, 3, 1, 2])

        decode = tf.reshape(NCHW_conv_transpose4, [-1, 1, 33600])
        return decode