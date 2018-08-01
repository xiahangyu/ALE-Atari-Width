import tensorflow as tf
import numpy as np
from AEModel import AEModel
import matplotlib.pyplot as plt

ae = AEModel()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "./freeway/cs/ckpt/original/submean/model")

#test data
n_test_x = BATCH_SIZE
test_x = np.zeros((n_test_x, 33600))

screen_dir = "../backup/screens/freeway/original/matrix/"
def loadScreen():
    for i in range(0, n_test_x):
        path = screen_dir + str(4480 + i + 1) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            pixels = data[:-1]
            pixels = list(map(int, pixels))
            test_x[i] = np.array(pixels)
loadScreen()

x_hat = sess.run(ae.x_hat, feed_dict={ae.x: test_x})
# hidden = sess.run(ae.hidden, feed_dict={ae.x: test_x})
# print(np.sum(hidden>=256), np.sum(hidden>=256)/hidden.size)
# print(np.sum(hidden>=512), np.sum(hidden>=512)/hidden.size)
# print(np.sum(hidden>=1024), np.sum(hidden>=1024)/hidden.size)

fig, axs = plt.subplots(2, 2, figsize=(210, 160), squeeze=False)
for example_i in range(2):
    axs[example_i][0].imshow(
        np.reshape(test_x[example_i], (210, 160)))
    axs[example_i][1].imshow(
        np.reshape(x_hat[example_i], (210, 160)))
fig.show()
plt.show()

sess.close()