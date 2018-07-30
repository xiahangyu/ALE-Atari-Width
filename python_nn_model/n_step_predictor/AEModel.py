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
        self.y = tf.placeholder(tf.float32, [None, NUM_STEP, 33600], name='y')
        self.n_step_acts = tf.placeholder(tf.float32, [None, NUM_STEP, 18], name='n_step_acts')
        self.one_step_act = tf.placeholder(tf.float32, [None,18], name='one_step_act')


        self.mean = tf.Variable(mean_img, trainable=False, dtype=tf.float32)
        self.x_mean = (self.x-self.mean)

        self.train_nn()
        self.one_step_pred_nn()

        self.merged = tf.summary.merge_all()


    def one_step_train(self, curr_k_screens, current_act):
        with tf.variable_scope("predict"):
            #encode
            encode, conv_shapes = layer.conv_encoder(curr_k_screens)

            #action_transform
            pred_encode, _ = layer.action_transform(encode, current_act)

            #decode
            pred_mean = layer.conv_decoder(encode, conv_shapes)
            pred = (pred_mean+self.mean)
            
            #next step input
            if K > 1:
                ns_ksub1_screens_indices = tf.constant([[i, k] for i in range(0, BATCH_SIZE) for k in range(1, K) ]) 
                ns_ksub1_screens = tf.gather_nd(curr_k_screens, ns_ksub1_screens_indices)
                ns_ksub1_screens = tf.reshape(ns_ksub1_screens, [-1, K-1, 33600])
                ns_k_screens = tf.concat([ns_ksub1_screens, pred_mean], 1)
            else:
                ns_k_screens = curr_k_screens
        return pred, ns_k_screens 


    def train_nn(self):
        y_hat_list = []
        with tf.variable_scope("train_nn"):
            curr_k_screens = self.x_mean
            for step in range(0, NUM_STEP):
                current_acts_indices = tf.constant([[i, step] for i in range(BATCH_SIZE)]) 
                current_acts = tf.gather_nd(self.n_step_acts, current_acts_indices)  

                y_hat, next_k_screens = self.one_step_train(curr_k_screens, current_act)
                curr_k_screens = next_k_screens
                y_hat_list.append(y_hat)

        with tf.variable_scope("cost"):
            self.y_hat = tf.concat(y_hat_list, axis = 1)
            self.cost = tf.reduce_mean(tf.square(self.y_hat - self.y), name="cost")

        with tf.variable_scope("optimize"):
            learning_rate = 0.0001
            self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)


    #predict the next screen
    def one_step_pred_nn(self):
        curr_k_screens = self.x_mean

        #encode
        encode, conv_shapes = layer.conv_encoder(curr_k_screens)
        self.hidden1 = tf.cast(tf.round(encode), tf.int32, name="hidden1")

        #action_transform
        pred_encode, encode_factor = layer.action_transform(encode, self.one_step_act)
        self.hidden2 = tf.cast(tf.round(encoder_factor), tf.int32, name="hidden2")
        self.hidden3 = tf.cast(tf.round(pred_encode), tf.int32, name="hidden3")

        #decode
        pred = layer.conv_decoder(pred_encode, conv_shapes)
        self.pred = tf.cast(tf.round(pred), tf.int32, name = "pred")