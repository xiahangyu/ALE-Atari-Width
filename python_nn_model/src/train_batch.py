import tensorflow as tf
import numpy as np
from AEModel import AEModel

batch_size = 128

n_train_screens = 2432
screen_height = 210
screen_width = 160

dev_screens = np.zeros((batch_size, screen_height*screen_width))
n_dev_screens = batch_size


screen_dir = "./screens/alien/"
current_pos = 1
def nextBatch(dir):
    batch_screen = np.zeros((batch_size, screen_height*screen_width))
    
    global current_pos
    for i in range(0, batch_size):
        path = dir + str(current_pos + i) + ".matrix"
        with open(path, "r") as f:
            pixels = f.read().split(' ')[:-1]
            pixels = list(map(int, pixels))
            batch_screen[i] = np.array(pixels)

    current_pos += batch_size
    if current_pos >= n_train_screens:
        current_pos = 1
    return batch_screen

def train():
    ae = AEModel()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  # max_to_keep=1
    writer = tf.summary.FileWriter("./AE_nn_log", sess.graph)

    print("training...")
    # Fit all training data
    n_epochs = 50
    for epoch_i in range(n_epochs):
        for batch_i in range(n_train_screens // batch_size):
            print(batch_i)
            batch_xs = nextBatch(screen_dir)
            sess.run(ae.optimizer, feed_dict={ae.x: batch_xs})
        print(epoch_i, sess.run(ae.cost, feed_dict={ae.x: batch_xs}), sess.run(ae.cost, feed_dict={ae.x: dev_screens}))
        summary, cost = sess.run([ae.merged, ae.cost], feed_dict={ae.x: dev_screens})
        writer.add_summary(summary, epoch_i)
    saver.save(sess, './ckpt/model')

    writer.close()
    sess.close()


if __name__ == '__main__':
    for i in range(0, batch_size):
        path = screen_dir + str(n_train_screens + i) + ".matrix"
        with open(path, "r") as f:
            pixels = f.read().split(' ')[:-1]
            pixels = list(map(int, pixels))
            dev_screens[i] = np.array(pixels)
    train()
