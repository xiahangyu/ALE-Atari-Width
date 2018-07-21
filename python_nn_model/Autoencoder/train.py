import tensorflow as tf
import numpy as np
from AEModel import AEModel

n_train_screens = 3840
n_dev_screens = 800
screen_height = 210
screen_width = 160
train_screens = np.zeros((n_train_screens, screen_height*screen_width))
dev_screens = np.zeros((n_dev_screens, screen_height*screen_width))
screen_dir = "./screens/freeway/"


def loadData(dir):
    for i in range(n_train_screens):
        path = dir + str(i+1) + ".matrix"
        with open(path, "r") as f:
            pixels = f.read().split(' ')[:-1]
            pixels = list(map(int, pixels))
            train_screens[i] = np.array(pixels)
    for i in range(n_dev_screens):
        path = dir + str(n_train_screens+i+1) + ".matrix"
        with open(path, "r") as f:
            pixels = f.read().split(' ')[:-1]
            pixels = list(map(int, pixels))
            dev_screens[i] = np.array(pixels)


def train(mean_img):
    ae = AEModel(mean_img=mean_img)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  # max_to_keep=1
    writer = tf.summary.FileWriter("./AE_nn_log", sess.graph)

    batch_size = 128
    # Fit all training data
    n_epochs = 1
    for epoch_i in range(n_epochs):
        np.random.shuffle(train_screens)
        for batch_i in range(n_train_screens // batch_size):
            batch_xs = train_screens[batch_i * batch_size: (batch_i + 1) * batch_size]
            # train = np.array([img - mean_img for img in batch_xs])
            sess.run(ae.optimizer, feed_dict={ae.x: batch_xs})
        print(epoch_i, sess.run(ae.cost, feed_dict={ae.x: train_screens}), sess.run(ae.cost, feed_dict={ae.x: dev_screens}))
        summary, cost = sess.run([ae.merged, ae.cost], feed_dict={ae.x: train_screens})
        writer.add_summary(summary, epoch_i)
    saver.save(sess, './ckpt/model')

    writer.close()
    sess.close()


if __name__ == '__main__':
    # load atari game screens
    loadData(screen_dir)
    mean_img = np.mean(train_screens, axis=0)
    mean_img = np.reshape(mean_img, [1, 33600])
    train(mean_img)
