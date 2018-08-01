import numpy as np

n_screens = 4645    #
screens = np.zeros((n_screens, 33600))

screen_dir = "./freeway/subtracted/matrix/"     #
def loadScreen():
    for i in range(0, n_screens):
        path = screen_dir + str(i + 1) + ".matrix"
        with open(path, "r") as f:
            data = f.read().split(' ')
            pixels = data[:-1]
            pixels = list(map(int, pixels))
            screens[i] = np.array(pixels)

fp = "./freeway/subtracted/mean.matrix"     #
fp = "./freeway/subtracted/submean.matrx"
def save_mean():
    mean_img = np.around(np.mean(screens, 0))
    mean_img = mean_img.astype(int)
    with open(fp, "w") as f:
        for i in range(33600):
            f.write(str(mean_img[i])+" ")
    print("cost:",np.mean(np.square(screens-mean_img)))

    submean_img = np.
    with open(fp, "w") as f:
        for i in range(33600):
            f.write(str(mean_img[i])+" ")

if __name__ == '__main__':
    loadScreen()
    save_mean()

