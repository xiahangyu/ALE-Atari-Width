import tensorflow as tf
import math
import numpy as np
from libs.activations import lrelu

screen_height = 210
screen_width = 160


class AEModel(object):
    def __init__(self,
                 input_shape=[None, 33600],
                 n_filters=[1, 16, 32, 64],
                 filter_sizes=[3, 3, 3, 3],
                 mean_img=np.zeros([1, 33600])) :
        tf.reset_default_graph()

        self.x = tf.placeholder(tf.float32, input_shape, name='x')
        self.mean = tf.Variable(mean_img, trainable=False, dtype=tf.float32)
        x_tensor = tf.subtract(self.x, self.mean)

        # %%
        # ensure 2-d is converted to square tensor.
        if len(x_tensor.get_shape()) == 2:
            x_tensor = tf.reshape(x_tensor, [-1, screen_height, screen_width, n_filters[0]])
        else:
            raise ValueError('Unsupported input dimensions')
        current_input = x_tensor

        # %%
        # Build the encoder
        encoder = []
        shapes = []
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes.append(current_input.get_shape().as_list())
            w = tf.Variable(
                tf.random_uniform([
                    filter_sizes[layer_i],
                    filter_sizes[layer_i],
                    n_input, n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            encoder.append(w)
            output = lrelu(tf.add(tf.nn.conv2d(current_input, w, strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output
        # %%
        # store the latent representation
        self.z = tf.sigmoid(current_input)
        self.z = tf.multiply(self.z, 255)
        self.z = tf.cast(tf.reshape(self.z, [-1, 34560]), tf.int32)
        tf.summary.histogram("Hidden_hist", self.z)
        encoder.reverse()
        shapes.reverse()

        # %%
        # Build the decoder using the same weights
        for layer_i, shape in enumerate(shapes):
            w = encoder[layer_i]
            b = tf.Variable(tf.zeros([w.get_shape().as_list()[2]]))
            output = tf.sigmoid(tf.add(
                tf.nn.conv2d_transpose(
                    current_input, w,
                    tf.stack([tf.shape(x_tensor)[0], shape[1], shape[2], shape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        # %%
        # now have the reconstruction through the network
        y_tensor = current_input
        self.y = tf.cast(tf.add(tf.reshape(y_tensor, [-1, 33600]), self.mean, name="y"), tf.int32)
        # cost function measures pixel-wise difference
        self.cost = tf.reduce_sum(tf.square(y_tensor - x_tensor), name="cost")
        tf.summary.scalar("Cost_scalar", self.cost)
        tf.summary.histogram("Cost_hist", self.cost)

        # %%
        learning_rate = 0.01
        self.optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(self.cost)

        self.merged = tf.summary.merge_all()