import tensorflow as tf
import numpy as np
from AEModel import AEModel
import os

ae = AEModel()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "./python_nn_model/Autoencoder/ckpt/model")

def predict(screen_values):
    screen = np.array(screen_values)
    screen = np.reshape(screen, [1, 33600])
    hidden_state = sess.run(ae.hidden, feed_dict={ae.x: screen})
    hidden_state = np.reshape(hidden_state, [128])
    return [v for v in hidden_state]

def close():
    sess.close()