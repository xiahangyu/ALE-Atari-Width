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
        self.one_step_act = tf.placeholder(tf.float32, [None, 18], name='one_step_act')

        self.mean = tf.Variable(mean_img, trainable=False, dtype=tf.float32)
        self.x_mean = self.x-self.mean

        self.train_nn()
        self.one_step_pred_nn()

        self.merged = tf.summary.merge_all()


    def predict(self, current_k_screens, current_act):
        with tf.variable_scope("predict"):
            #encode
            encode, conv_shapes = layer.conv_encoder(current_k_screens)

            #action_transform
            pred_encode = layer.action_transform(encode, current_act)

            #decode
            pred_mean = layer.conv_decoder(pred_encode, conv_shapes)
            pred = pred_mean + self.mean
            
            #next step input
            ns_ksub1_indices = tf.constant([[i, k] for i in range(0, BATCH_SIZE) for k in range(1, K) ]) 
            ns_ksub1_screens = tf.gather_nd(current_k_screens, ns_ksub1_indices)
            ns_ksub1_screens = tf.reshape(ns_ksub1_screens, [-1, K-1, 33600])
            ns_k_screens = tf.concat([ns_ksub1_screens, pred_mean], 1)
        return pred, ns_k_screens


    def train_nn(self):
        y_hat_list = []
        with tf.variable_scope("train_nn"):
            current_k_screens = self.x_mean
            for step in range(0, NUM_STEP):
                current_act_indices = tf.constant([[i, step] for i in range(BATCH_SIZE)]) 
                current_act = tf.gather_nd(self.n_step_acts, current_act_indices)  #act [None, 18]

                y_pred, next_k_screens = self.predict(current_k_screens, current_act)
                current_k_screens = next_k_screens
                y_hat_list.append(y_pred)

        with tf.variable_scope("cost"):
            self.y_hat = tf.concat(y_hat_list, axis = 1)
            self.cost = tf.reduce_mean(tf.square(self.y_hat - self.y), name="cost")
            tf.summary.scalar("cost", self.cost)

        with tf.variable_scope("optimize"):
            learning_rate = 0.01
            self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)


    #predict the next screen
    def one_step_pred_nn(self):
        current_k_screens = self.x_mean
        encode, conv_shapes = layer.conv_encoder(current_k_screens)
        pred_encode = layer.action_transform(encode, self.one_step_act)
        pred_mean = layer.conv_decoder(pred_encode, conv_shapes)
        pred = pred_mean + self.mean
        self.pred = tf.cast(pred, tf.int32, name = "pred")
