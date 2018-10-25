import tensorflow as tf
import numpy as np
import constants as const
from AEModel import AEModel

BATCH_SIZE = const.BATCH_SIZE
NUM_ACTIONS = const.NUM_ACTIONS
K = const.K   
NUM_STEP = const.NUM_STEP
T = const.T  

#    raw data    
#training data
n_train_screens =4464+1    #
train_screens = np.zeros((BATCH_SIZE*K+NUM_STEP, 33600))
train_screens_acts = np.zeros((BATCH_SIZE*K+NUM_STEP, NUM_ACTIONS))    #one hot actions, [0...17]
#develop data
n_dev_screens = 496
dev_screens = np.zeros((BATCH_SIZE*K+NUM_STEP, 33600))
dev_screens_acts = np.zeros((BATCH_SIZE*K+NUM_STEP, NUM_ACTIONS))

# structured data 
#training data
train_x = np.zeros((BATCH_SIZE, K, 33600))
train_y = np.zeros((BATCH_SIZE, NUM_STEP, 33600))
train_acts = np.zeros((BATCH_SIZE, NUM_STEP, NUM_ACTIONS))
#develop data
dev_x = np.zeros((BATCH_SIZE, K, 33600))
dev_y = np.zeros((BATCH_SIZE, NUM_STEP, 33600))
dev_acts = np.zeros((BATCH_SIZE, NUM_STEP, NUM_ACTIONS))

screen_dir = "../../../../screens/freeway/subtracted/matrix_act/"

current_pos = 1
def nextBatch():
    global current_pos
    #read in screen_act
    for i in range(0, BATCH_SIZE*K+NUM_STEP):
        path = screen_dir + str(current_pos + i) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            act = int(data[-2])
            train_screens_acts[i][act] = 1

            pixels = data[:-2]
            pixels = list(map(int, pixels))
            train_screens[i] = np.array(pixels)

    #build training date
    for i in range(0, BATCH_SIZE):
        train_x[i] = train_screens[i*K:i*K+K]
        train_y[i] = train_screens[i*K+K:i*K+T]
        train_acts[i] = train_screens_acts[i*K+K-1:i*K+T-1]

    current_pos += BATCH_SIZE*K
    if current_pos >= n_train_screens:
        current_pos = 1

dev_current_pos = n_train_screens
def nextDevBatch():
    global dev_current_pos
    #read in screen_act
    # print(dev_current_pos)
    for i in range(0, BATCH_SIZE*K+NUM_STEP):
        # print(" ", dev_current_pos + i, i)
        path = screen_dir + str(dev_current_pos + i) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            act = int(data[-2])
            dev_screens_acts[i][act] = 1

            pixels = data[:-2]
            pixels = list(map(int, pixels))
            dev_screens[i] = np.array(pixels)

    #build training date
    for i in range(0, BATCH_SIZE):
        dev_x[i] = dev_screens[i*K:i*K+K]
        dev_y[i] = dev_screens[i*K+K:i*K+T]
        dev_acts[i] = dev_screens_acts[i*K+K-1:i*K+T-1]

    dev_current_pos += BATCH_SIZE*K
    if dev_current_pos >= n_train_screens+n_dev_screens:
        dev_current_pos = n_train_screens

def train():
    n_epochs = 200
    ae = AEModel() 

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
                sess.run(ae.optimizer, feed_dict={ae.x: train_x, ae.y: train_y, ae.n_step_acts: train_acts})
            
            cost = 0
            for batch_i in range(n_dev_screens // BATCH_SIZE):
                nextDevBatch()
                cost += sess.run(ae.cost, feed_dict={ae.x: dev_x, ae.y: dev_y ,ae.n_step_acts: dev_acts})
            cost = cost/(n_dev_screens // BATCH_SIZE)
            print("epoch", epoch_i+1, ":",cost)

            cost_summary.value[0].simple_value = cost
            summary_writer.add_summary(cost_summary, epoch_i+1)
            summary_writer.flush()

        c_saver.save(sess, "./ckpt/graph.ckpt")
        tf.train.write_graph(sess.graph_def, './ckpt/', 'graph.pbtxt', as_text=True)


if __name__ == '__main__':
    # loadDevData()
    train()