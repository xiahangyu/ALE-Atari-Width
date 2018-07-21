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

'''
n_train_screens = 4444
n_dev_screens = 1
screen_height = 210
screen_width = 160
dev_screens = np.zeros((n_dev_screens, screen_height*screen_width))
screen_dir = "./screens/freeway/"


def loadData(dir):
    for i in range(n_dev_screens):
        path = dir + str(n_train_screens+i+1) + ".matrix"
        with open(path, "r") as f:
            pixels = f.read().split(' ')[:-1]
            pixels = list(map(int, pixels))
            dev_screens[i] = np.array(pixels)


print(predict(dev_screens[0]))
'''