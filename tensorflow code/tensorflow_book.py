import tensorflow as tf
import numpy as np


g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32,
                       shape=(None, 2, 3),
                       name='input_x')
    x2 = tf.reshape(x, shape=(-1, 6),
                    name='x2')
    xsum = tf.reduce_sum(x, axis=0, name='col_sum')

    xmean = tf.reduce_mean(x2, axis=0, name='col_mean')

with tf.Session(graph=g) as sess:
    x_array = np.arange(18).reshape(3, 2, 3)
    print(x_array.shape, '\n', x_array)
    print('reshaped\n', sess.run(x2, feed_dict={x: x_array}))
    print('col sum\n', sess.run(xsum, feed_dict={x: x_array}))
    print('col sum\n', sess.run(xmean, feed_dict={x: x_array})
