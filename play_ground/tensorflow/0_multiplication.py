import tensorflow as tf
import numpy as np

## ---------------------------------------------------------------------
## Multiplication
## ---------------------------------------------------------------------

## Create placeholder to feed in the data
x = tf.placeholder("float")
w = tf.placeholder("float")

## Create symbolic variable
y = tf.mul(x, w)
y_mat = tf.matmul(x, w)

## Multiply
with tf.Session() as sess:
    ##  scalar multiplication
    print("%f should equal to 2.0" % sess.run(y, feed_dict = {x: 1, w: 2}))
    print("%f should equal to 8.0" % sess.run(y, feed_dict = {x: 2, w: 4}))
    ## Matrix multiplication
    print("%f should equal to 12.0" %
          sess.run(y_mat, feed_dict = {x: np.array([[2, 3]]),
                                       w: np.array([[3], [2]])}))
    print("%f should equal to 66.0" %
          sess.run(y_mat, feed_dict = {x: np.array([[5, 8]]),
                                       w: np.array([[10], [2]])}))



