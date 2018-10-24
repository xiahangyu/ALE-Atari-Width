import tensorflow as tf
import numpy as np
import constants as const

class layer():

    #Inputs:
    #   x: tensor, [batch_size, n_steps * n_input]
    #   n_hidden: float
    #Outputs:
    #   encode: tensor, [1, batch_size, n_hidden]
    #   states: tensor, [n_hidden]
    def rnn_encoder(x, n_hidden):
        x = tf.reshape(x, [-1, 210, 160])
        with tf.variable_scope("rnn_encoder"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, reuse=tf.AUTO_REUSE)
            outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, dtype=tf.float32)
            outputs = tf.transpose(outputs,[1,0,2])    #tensor, [n_steps, batch_size, n_input]
            encode = tf.reshape(outputs[-1], [-1, 160])
        return encode, states


    #Inputs:
    #   encode : tensor, [None, 160]
    #   initial_states : tensor, [n_hidden]
    #   n_hidden : float
    #Outputs:
    #   decode : tensor, [None, 33600]
    def rnn_decoder(encode, states, n_hidden):
        with tf.variable_scope("rnn_decoder"):
            inputs = tf.reshape(encode, [-1, 1, 160])

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, reuse=tf.AUTO_REUSE)
            outputs = []
            for i in range(210):
                inputs, states = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=inputs,
                                                    initial_state = states, 
                                                    dtype=tf.float32)

                outputs.append(inputs)  #a list of [None, 1, 160]
            decode = tf.concat(outputs, axis = 1)   #tensor, [None, 210, 160]
            decode = tf.reshape(decode, [-1, 33600])
        return decode