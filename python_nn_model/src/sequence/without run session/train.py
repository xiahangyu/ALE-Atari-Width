import tensorflow as tf
import numpy as np
import constants as const
from AEModel import AEModel

batch_size = const.BATCH_SIZE

NUM_ACTIONS = const.NUM_ACTIONS
K = const.K   #num of frames to look back
NUM_STEP = const.NUM_STEP    #num of frames to predict
T = const.T  # K + NUM_STEP

#    raw data     #
#training data
n_train_screens = 128   #2816
train_screens = np.zeros((n_train_screens, 33600))
train_screens_act = np.zeros((n_train_screens, NUM_ACTIONS))    #one hot actions, [0...17]
#develop data
n_dev_screens = 128
dev_screens = np.zeros((n_dev_screens, 33600))
dev_screens_act = np.zeros((n_dev_screens, NUM_ACTIONS))

# structured data #
#training data
train_x = np.zeros((n_train_screens, K, 33600))
train_y = np.zeros((n_train_screens, NUM_STEP, 33600))
train_act = np.zeros((n_train_screens, NUM_STEP, NUM_ACTIONS))
#develop data
dev_x = np.zeros((n_dev_screens, K, 33600))
dev_y = np.zeros((n_dev_screens, NUM_STEP, 33600))
dev_act = np.zeros((n_dev_screens, NUM_STEP, NUM_ACTIONS))


screen_dir = "./screens/alien/matrix_act/"
def loadScreen():
    print("load screens...")
    for i in range(0, n_train_screens):
        path = screen_dir + str(i + 1) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            act = int(data[-2])
            train_screens_act[i][act] = 1

            pixels = data[:-2]
            pixels = list(map(int, pixels))
            train_screens[i] = np.array(pixels)

    for i in range(0, n_dev_screens):
        path = screen_dir + str(n_train_screens + i + 1) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')

            act = int(data[-2])
            dev_screens_act[i][act] = 1 

            pixels = data[:-2]
            pixels = list(map(int, pixels))
            dev_screens[i] = np.array(pixels)
    print("screen loaded")


def buildData():
    print("build data...")
    for i in range(0, n_train_screens-T):
        train_x[i] = train_screens[i:i+K]
        train_y[i] = train_screens[i+K:i+T]
        train_act[i] = train_screens_act[i+K-1:i+T-1]

    for i in range(0, n_dev_screens-T):
        dev_x[i] = dev_screens[i:i+K]
        dev_y[i] = dev_screens[i+K:i+T]
        dev_act[i] = dev_screens_act[i+K-1:i+T-1]
    print("data builed")


n_batch = n_train_screens//batch_size
def train(mean_img):
    ae = AEModel(mean_img = mean_img)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    py_saver = tf.train.Saver()
    c_saver = tf.train.Saver(tf.global_variables())

    writer = tf.summary.FileWriter("./AE_nn_log", sess.graph)

    n_epochs = 5
    for epoch_i in range(n_epochs):
        for batch_i in range(n_batch):
            print(batch_i)
            batch_x = train_x[batch_i*batch_size : batch_i*batch_size + batch_size]
            batch_y = train_y[batch_i*batch_size : batch_i*batch_size + batch_size]
            batch_act = train_act[batch_i*batch_size : batch_i*batch_size + batch_size]
            sess.run(ae.optimizer, feed_dict={ae.x: batch_x, ae.y: batch_y, ae.n_step_acts: batch_act})
        print(epoch_i, sess.run(ae.cost, feed_dict={ae.x: dev_x, ae.y: dev_y, ae.n_step_acts: dev_act}))
        #summary, cost = sess.run([ae.merged, ae.cost], feed_dict={ae.x: dev_screens})
        #writer.add_summary(summary, epoch_i)
    py_saver.save(sess, './ckpt/model')
    c_saver.save(sess, "c_ckpt/graph.ckpt")
    tf.train.write_graph(sess.graph_def, 'c_ckpt/', 'graph.pbtxt', as_text=True)

    writer.close()
    sess.close()


if __name__ == '__main__':
    loadScreen()
    buildData()
    mean_img = np.mean(train_screens, 0)
    train(mean_img)
