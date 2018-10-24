import tensorflow as tf
import numpy as np
import constants as const
from AEModel import AEModel

BATCH_SIZE = const.BATCH_SIZE

#training data
n_train_screens = 4928 #4960 32
train_screens = np.zeros((BATCH_SIZE, 33600))
#develop data
dev_screens = np.zeros((BATCH_SIZE, 33600))


# mean_img = np.zeros((33600))
# mean_img_path = "../screens/freeway/subtracted/mean.matrix"   #
# def loadMeanImg():
#     with open(mean_img_path, "r") as f:
#         data = f.read().split(' ')
#         pixels = data[:-1]
#         pixels = list(map(int, pixels))
#         mean_img = np.array(pixels)

screen_dir = "../../../../screens/freeway/subtracted/matrix/"  #
def loadDevData():
    for i in range(0, BATCH_SIZE):
        path = screen_dir + str(n_train_screens + i) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            pixels = data[:-1]
            pixels = list(map(int, pixels))
            dev_screens[i] = np.array(pixels)


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


def train():
    ae = AEModel() #mean_img = mean_img
    sess = tf.Session()
    py_saver = tf.train.Saver()
    c_saver = tf.train.Saver(tf.global_variables())

    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter("./log", sess.graph)

    n_epochs = 200
    print("training...")
    for epoch_i in range(n_epochs):
        for batch_i in range(n_train_screens // BATCH_SIZE):
            nextBatch()
            sess.run(ae.optimizer, feed_dict={ae.x: train_screens})
        summary_str, cost = sess.run([ae.merged, ae.cost], feed_dict={ae.x: dev_screens})
        print("epoch", epoch_i+1, ":",cost)

        summary_writer.add_summary(summary_str, epoch_i+1)
        summary_writer.flush()
        # if epoch_i > 90 and cost<2:
        #     break


    py_saver.save(sess, './ckpt/model')
    c_saver.save(sess, "./ckpt/graph.ckpt")
    tf.train.write_graph(sess.graph_def, './ckpt/', 'graph.pbtxt', as_text=True)

    sess.close()


if __name__ == '__main__':
    #loadMeanImg()
    loadDevData()
    train()