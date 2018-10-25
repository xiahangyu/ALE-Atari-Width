import tensorflow as tf
import numpy as np
import constants as const
from AEModel import AEModel

BATCH_SIZE = const.BATCH_SIZE

#training data
n_train_screens = 4480
train_screens = np.zeros((BATCH_SIZE, 33600))

#develop data
n_dev_screens = 512
dev_screens = np.zeros((BATCH_SIZE, 33600))

screen_dir = "../../../../screens/freeway/subtracted/matrix/"  #

current_pos = 1
def nextBatch():
    global current_pos
    for i in range(0, BATCH_SIZE):
        path = screen_dir + str(current_pos + i) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            pixels = data[:-1]
            pixels = list(map(int, pixels))
            train_screens[i] = np.array(pixels)

    current_pos += BATCH_SIZE
    if current_pos >= n_train_screens:
        current_pos = 1

dev_current_pos = n_train_screens
def nextDevBatch():
    global dev_current_pos
    for i in range(0, BATCH_SIZE):
        path = screen_dir + str(dev_current_pos + i) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            pixels = data[:-1]
            pixels = list(map(int, pixels))
            dev_screens[i] = np.array(pixels)

    dev_current_pos += BATCH_SIZE
    if dev_current_pos >= n_train_screens+n_dev_screens:
        dev_current_pos = n_train_screens

def train():
    n_epochs = 200
    ae = AEModel() #mean_img = mean_img

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        c_saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter("./log", sess.graph)
        
        cost_summary = tf.Summary()
        cost_summary.value.add(tag="cost", simple_value=None)

        print("training...")
        for epoch_i in range(n_epochs):
            for batch_i in range(n_train_screens // BATCH_SIZE):
                nextBatch()
                sess.run(ae.optimizer, feed_dict={ae.x: train_screens})

            cost = 0
            for batch_i in range(n_dev_screens // BATCH_SIZE):
                nextDevBatch()
                cost += sess.run(ae.cost, feed_dict={ae.x: dev_screens})
            cost = cost/(n_dev_screens // BATCH_SIZE)
            print("epoch", epoch_i+1, ":",cost)

            cost_summary.value[0].simple_value = cost
            summary_writer.add_summary(cost_summary, epoch_i+1)
            summary_writer.flush()

        c_saver.save(sess, "./ckpt/graph.ckpt")
        tf.train.write_graph(sess.graph_def, './ckpt/', 'graph.pbtxt', as_text=True)

if __name__ == '__main__':
    train()