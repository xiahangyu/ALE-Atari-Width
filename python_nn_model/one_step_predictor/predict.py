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
test_x = np.zeros((BATCH_SIZE, 33600))
test_y = np.zeros((BATCH_SIZE, 33600))
test_act = np.zeros((BATCH_SIZE, 18))

screen_act_dir = "../screens/freeway/subtracted/matrix_act/"  #
def loadTestData():
    for i in range(0, BATCH_SIZE+1):
        path = screen_act_dir + str(4000 + i) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            act = int(data[-2])
            test_act[i][act] = 1

            pixels = data[:-2]
            pixels = list(map(int, pixels))
            if i < BATCH_SIZE:
                test_x[i] = np.array(pixels)
            if i > 0:
                test_y[i-1] = np.array(pixels)


y_hat = sess.run(ae.y_hat, feed_dict={ae.x: test_x, ae.y: test_y, ae.act: test_act})

n_examples = 2
fig, axs = plt.subplots(n_examples, 2, figsize=(210, 160), squeeze=False)
for example_i in range(n_examples):
    axs[example_i][0].imshow(
        np.reshape(test_y[49][0], (210, 160)))
    axs[example_i][1].imshow(
        np.reshape(y_hat[49][0], (210, 160)))
fig.show()
plt.show()

sess.close()