
import matplotlib.pyplot as plt

ae = AEModel()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "./freeway/ns/ckpt/subtracted/K=3, N=1/model")

test_screens = np.zeros((BATCH_SIZE+T, 33600))
test_screens_acts = np.zeros((BATCH_SIZE+T, NUM_ACTIONS))
#test data
test_x = np.zeros((BATCH_SIZE, K, 33600))
test_y = np.zeros((BATCH_SIZE, 33600))
test_acts = np.zeros((BATCH_SIZE, 18))

screen_act_dir = "../backup/screens/freeway/subtracted/matrix_act/"  #
def loadTestData():
    for i in range(0, BATCH_SIZE+T):
        path = screen_act_dir + str(1950 + i) + ".matrix"  #
        with open(path, "r") as f:
            data = f.read().split(' ')
            act = int(data[-2])
            test_screens_acts[i][act] = 1

            pixels = data[:-2]
            pixels = list(map(int, pixels))
            test_screens[i] = np.array(pixels)

    #build training date
    for i in range(0, BATCH_SIZE):
        test_x[i] = test_screens[i:i+K]
        test_y[i] = test_screens[i+K]
        test_acts[i] = test_screens_acts[i+K-1]
loadTestData()
        
pred, hidden1, hidden2, hidden3 = sess.run([ae.pred, ae.hidden1, ae.hidden2, ae.hidden3], feed_dict={ae.x: test_x, ae.one_step_act: test_acts})
print("hidden1")
print(np.max(hidden1))
print(np.sum(hidden1>=256), np.sum(hidden1>=256)/hidden1.size)
print(np.sum(hidden1>=512), np.sum(hidden1>=512)/hidden1.size)
print(np.sum(hidden1>=1024), np.sum(hidden1>=1024)/hidden1.size)

print("hidden2")
print(np.max(hidden1))
print(np.sum(hidden2>=256), np.sum(hidden2>=256)/hidden2.size)
print(np.sum(hidden2>=512), np.sum(hidden2>=512)/hidden2.size)
print(np.sum(hidden2>=1024), np.sum(hidden2>=1024)/hidden2.size)

print("hidden3")
print(np.max(hidden1))
print(np.sum(hidden3>=256), np.sum(hidden3>=256)/hidden3.size)
print(np.sum(hidden3>=512), np.sum(hidden3>=512)/hidden3.size)
print(np.sum(hidden3>=1024), np.sum(hidden3>=1024)/hidden3.size)

n_examples = 2
fig, axs = plt.subplots(n_examples, 2, figsize=(210, 160), squeeze=False)
for example_i in range(n_examples):
    axs[example_i][0].imshow(
        np.reshape(test_y[example_i], (210, 160)))
    axs[example_i][1].imshow(
        np.reshape(pred[example_i][0], (210, 160)))
fig.show()
plt.show()

sess.close()