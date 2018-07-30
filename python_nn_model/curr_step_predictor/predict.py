import tensorflow as tf
import numpy as np
from AEModel import AEModel
import matplotlib.pyplot as plt

ae = AEModel()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "./ckpt/model")

#test data
n_test_x = BATCH_SIZE
test_x = np.zeros((n_test_x, 33600))

screen_dir = "../screens/freeway/subtracted/matrix/"
def loadScreen():
    for i in range(0, n_test_x):
        path = screen_dir + str(3840 + i + 1) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            pixels = data[:-1]
            pixels = list(map(int, pixels))
            test_x[i] = np.array(pixels)
loadScreen()

x_hat = sess.run(ae.x_hat, feed_dict={ae.x: test_x})

n_examples = 2
fig, axs = plt.subplots(n_examples, 2, figsize=(210, 160), squeeze=False)
for example_i in range(n_examples):
    axs[example_i][0].imshow(
        np.reshape(test_x[49][0], (210, 160)))
    axs[example_i][1].imshow(
        np.reshape(x_hat[49][0], (210, 160)))
fig.show()
plt.show()

sess.close()