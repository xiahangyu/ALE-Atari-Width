import tensorflow as tf
import numpy as np
from AEModel import AEModel

screen_height = 210
screen_width = 160
batch_size = 128

n_train_screens = 2432
train_screens = np.zeros((n_train_screens, screen_height*screen_width))
n_dev_screens = batch_size
dev_screens = np.zeros((n_dev_screens, screen_height*screen_width))

screen_dir = "./screens/alien/"
def loadData(dir):
    for i in range(0, n_train_screens):
        path = dir + str(i + 1) + ".matrix"
        with open(path, "r") as f:
            pixels = f.read().split(' ')[:-1]
            pixels = list(map(int, pixels))
            train_screens[i] = np.array(pixels)

    for i in range(0, n_dev_screens):
        path = dir + str(n_train_screens + i + 1) + ".matrix"
        with open(path, "r") as f:
            pixels = f.read().split(' ')[:-1]
            pixels = list(map(int, pixels))
            dev_screens[i] = np.array(pixels)


n_batch = n_train_screens//batch_size
def train(mean_img):
    ae = AEModel(mean_img)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("./AE_nn_log", sess.graph)

    n_epochs = 50
    for epoch_i in range(n_epochs):
        for batch_i in range(n_batch):
            print(batch_i)
            batch_xs = train_screens[batch_i*batch_size : batch_i*batch_size + batch_size]
            sess.run(ae.optimizer, feed_dict={ae.x: batch_xs})
        print(epoch_i, sess.run(ae.cost, feed_dict={ae.x: dev_screens}))
        summary, cost = sess.run([ae.merged, ae.cost], feed_dict={ae.x: dev_screens})
        writer.add_summary(summary, epoch_i)
    saver.save(sess, './ckpt/model')

    writer.close()
    sess.close()


if __name__ == '__main__':
    loadData(screen_dir)
    mean_img = np.mean(train_screens, 0)
    train(mean_img)
