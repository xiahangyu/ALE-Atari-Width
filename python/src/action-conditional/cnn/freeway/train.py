import tensorflow as tf
import numpy as np
import constants as const
from AEModel import AEModel

BATCH_SIZE = const.BATCH_SIZE

#training data
n_train_screens = 4928    #4960 32, 
train_x = np.zeros((BATCH_SIZE, 33600))
train_y = np.zeros((BATCH_SIZE, 33600))
train_act = np.zeros((BATCH_SIZE, 18))

#develop data
dev_x = np.zeros((BATCH_SIZE, 33600))
dev_y = np.zeros((BATCH_SIZE, 33600))
dev_act = np.zeros((BATCH_SIZE, 18))

screen_act_dir = "../../../../screens/freeway/subtracted/matrix_act/"  #
def loadDevData():
    for i in range(0, BATCH_SIZE+1):
        path = screen_act_dir + str(n_train_screens + i) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            act = int(data[-2])

            pixels = data[:-2]
            pixels = list(map(int, pixels))
            if i < BATCH_SIZE:
                dev_x[i] = np.array(pixels)
                dev_act[i][act] = 1
            if i > 0:
                dev_y[i-1] = np.array(pixels)


current_pos = 1
def nextBatch():
    global current_pos
    for i in range(0, BATCH_SIZE+1):
        path = screen_act_dir + str(current_pos + i) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            act = int(data[-2])

            pixels = data[:-2]
            pixels = list(map(int, pixels))
            if i < BATCH_SIZE:
                train_x[i] = np.array(pixels)
                train_act[i][act] = 1
            if i > 0:
                train_y[i-1] = np.array(pixels)

    current_pos += BATCH_SIZE
    if current_pos >= n_train_screens:
        current_pos = 1


def train():
    n_epochs = 200
    ae = AEModel() #mean_img = mean_img

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        c_saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter("./log", sess.graph)
        
        print("training...")
        for epoch_i in range(n_epochs):
            for batch_i in range(n_train_screens // BATCH_SIZE):
                nextBatch()
                sess.run(ae.optimizer, feed_dict={ae.x: train_x, ae.y: train_y, ae.act: train_act})
            summary_str, cost = sess.run([ae.merged, ae.cost], feed_dict={ae.x: dev_x, ae.y: dev_y, ae.act: dev_act})
            print("epoch", epoch_i+1, ":",cost)

            summary_writer.add_summary(summary_str, epoch_i+1)
            summary_writer.flush()

        c_saver.save(sess, "./ckpt/graph.ckpt")
        tf.train.write_graph(sess.graph_def, './ckpt/', 'graph.pbtxt', as_text=True)


if __name__ == '__main__':
    # loadMeanImg()
    loadDevData()
    train()