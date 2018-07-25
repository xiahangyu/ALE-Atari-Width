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
        self.x_mean = (self.x-self.mean)/255

        self.train_nn()
        self.one_step_pred_nn()


    def next_n_step(self, current_k_screens, step):
        with tf.variable_scope("step_n"):
            #encode
            encode, conv_shapes = layer.conv_encoder(current_k_screens)

            #action_transform
            current_act_indices = tf.constant([[i, step-1] for i in range(BATCH_SIZE)]) 
            current_act = tf.gather_nd(self.n_step_acts, current_act_indices)  #step_act [None, 18], corresponds to self.x [None, K, 33600]
            pred_encode = layer.action_transform(encode, current_act)

            #decode
            pred = layer.conv_decoder(pred_encode, conv_shapes)

            #true label of this step
            label_indices =  tf.constant([[i, step-1] for i in range(0, BATCH_SIZE)])
            label = tf.gather_nd(self.y, label_indices)
            label = tf.reshape(label, tf.stack([tf.shape(label)[0], 1, 33600]))

            #cost of this step
            cost = tf.reduce_mean(tf.square(pred - label))

            #next step input
            pred_mean = (pred-self.mean)/255
            ns_screen_indices = tf.constant([[i, k] for i in range(0, BATCH_SIZE) for k in range(1, K) ]) 
            ns_ksub1_screens = tf.gather_nd(current_k_screens, ns_screen_indices)
            ns_ksub1_screens = tf.reshape(ns_ksub1_screens, tf.stack([tf.shape(ns_ksub1_screens)[0], K-1, 33600]))
            ns_k_screens = tf.concat([ns_ksub1_screens, pred_mean], 1)

        return cost, ns_k_screens


    def train_nn(self):
        with tf.variable_scope("train_nn"):
            current_k_screens = self.x_mean
            cost_next_1, current_k_screens = self.next_n_step(current_k_screens, 1)
            cost_next_2, current_k_screens = self.next_n_step(current_k_screens, 2)
            cost_next_3, current_k_screens = self.next_n_step(current_k_screens, 3)

        with tf.variable_scope("cost"):
            total_cost = cost_next_1 + cost_next_2 + cost_next_3
            self.cost = tf.divide(total_cost, 3.0, name="cost")
            #print("cost:",self.cost.get_shape().as_list())
            #tf.summary.scalar("cost", self.cost)

        with tf.variable_scope("optimize"):
            learning_rate = 0.001
            self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)

        #self.merged = tf.summary.merge_all()


    #predict the next screen
    def one_step_pred_nn(self):
        current_k_screens = self.x_mean
        encode, conv_shapes = layer.conv_encoder(current_k_screens)
        pred_encode = layer.action_transform(encode, self.one_step_act)
        pred = layer.conv_decoder(pred_encode, conv_shapes)
        self.pred = tf.cast(pred, tf.int32, name = "pred")
