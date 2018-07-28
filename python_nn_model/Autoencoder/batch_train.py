import tensorflow as tf
import numpy as np
import constants as const
from AEModel import AEModel

BATCH_SIZE = const.BATCH_SIZE
NUM_ACTIONS = const.NUM_ACTIONS
K = const.K   #num of frames to look back
NUM_STEP = const.NUM_STEP    #num of frames to predict
T = const.T  # K + NUM_STEP

#    raw data    
#training data
n_train_screens = 2816  #
train_screens = np.zeros((BATCH_SIZE+T, 33600))
train_screens_act = np.zeros((BATCH_SIZE+T, NUM_ACTIONS))    #one hot actions, [0...17]
#develop data
dev_screens = np.zeros((BATCH_SIZE+T, 33600))
dev_screens_act = np.zeros((BATCH_SIZE+T, NUM_ACTIONS))

# structured data 
#training data
train_x = np.zeros((BATCH_SIZE, K, 33600))
train_y = np.zeros((BATCH_SIZE, NUM_STEP, 33600))
train_act = np.zeros((BATCH_SIZE, NUM_STEP, NUM_ACTIONS))
#develop data
dev_x = np.zeros((BATCH_SIZE, K, 33600))
dev_y = np.zeros((BATCH_SIZE, NUM_STEP, 33600))
dev_act = np.zeros((BATCH_SIZE, NUM_STEP, NUM_ACTIONS))

mean_img = np.zeros((33600))
mean_img_path = "./screens/alien/original/mean.matrix"   #
def loadMeanImg():
    with open(mean_img_path, "r") as f:
        data = f.read().split(' ')
        pixels = data[:-1]
        pixels = list(map(int, pixels))
        mean_img = np.array(pixels)

screen_dir = "./screens/alien/original/matrix_act/"  #
def loadDevData():
    for i in range(0, BATCH_SIZE+T):
        path = screen_dir + str(n_train_screens + i) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            act = int(data[-2])
            dev_screens_act[i][act] = 1

            pixels = data[:-2]
            pixels = list(map(int, pixels))
            dev_screens[i] = np.array(pixels)

    #build training date
    for i in range(0, BATCH_SIZE):
        dev_x[i] = dev_screens[i:i+K]
        dev_y[i] = dev_screens[i+K:i+T]
        dev_act[i] = dev_screens_act[i+K-1:i+T-1]

current_pos = 1
def nextBatch():
    global current_pos
    #read in screen_act
    for i in range(0, BATCH_SIZE+T):
        path = screen_dir + str(current_pos + i) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            act = int(data[-2])
            train_screens_act[i][act] = 1

            pixels = data[:-2]
            pixels = list(map(int, pixels))
            train_screens[i] = np.array(pixels)

    #build training date
    for i in range(0, BATCH_SIZE):
        train_x[i] = train_screens[i:i+K]
        train_y[i] = train_screens[i+K:i+T]
        train_act[i] = train_screens_act[i+K-1:i+T-1]

    current_pos += BATCH_SIZE
    if current_pos >= n_train_screens:
        current_pos = 1

def train():
    ae = AEModel(mean_img = mean_img)
    sess = tf.Session()
    py_saver = tf.train.Saver()
    c_saver = tf.train.Saver(tf.global_variables())
    writer = tf.summary.FileWriter("./AE_nn_log/", sess.graph)

    sess.run(tf.global_variables_initializer())

    n_epochs = 50
    early_stopping = 0
    last_cost = 0
    print("training...")
    for epoch_i in range(n_epochs):
        for batch_i in range(n_train_screens // BATCH_SIZE):
            print(batch_i)
            nextBatch()
            sess.run(ae.optimizer, feed_dict={ae.x: train_x, ae.y: train_y, ae.n_step_acts: train_act})
        print(epoch_i, ":", sess.run(ae.cost, feed_dict={ae.x: dev_screens}))
        summary, cost = sess.run([ae.merged, ae.cost], feed_dict={ae.x: dev_x, ae.y: dev_y ,ae.n_step_acts: dev_act})
        writer.add_summary(summary, epoch_i)
        writer.flush()
        print("epoch", epoch_i, ":", cost)
        if cost >= last_cost:
            early_stopping+=1
        last_cost = cost
        if early_stopping > 3:
            break

    py_saver.save(sess, './ckpt/model')
    c_saver.save(sess, "./c_ckpt/graph.ckpt")
    tf.train.write_graph(sess.graph_def, './c_ckpt/', 'graph.pbtxt', as_text=True)

    writer.close()
    sess.close()


if __name__ == '__main__':
    loadMeanImg()
    loadDevData()
    train()
