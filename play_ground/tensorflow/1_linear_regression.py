import tensorflow as tf
import numpy as np

## ---------------------------------------------------------------------
## Linear Regression
## ---------------------------------------------------------------------

## Simulate some data
weight = 0.5
bias = 2
sim_x = np.linspace(-1, 1, 101)
sim_y = bias + weight * sim_x + np.random.randn(*sim_x.shape) * 0.2

## Create placeholder to feed in the data
x = tf.placeholder("float")
y = tf.placeholder("float")
w = tf.Variable(0.0, name = "weights")
b = tf.Variable(0.0, name = "bias")

## Define model
def lm(x, w, b):
    return tf.mul(x, w) + b

## Create output
y_hat = lm(x, w, b)

## Define cost function
cost = tf.reduce_mean(tf.square(y - y_hat))

## Define training algorithm
train_alg = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    ## Initialize variables (in this case just variable W)
    tf.initialize_all_variables().run()
    for iter in range(1000):
        sess.run(train_alg, feed_dict = {x: sim_x, y: sim_y})
    print("Cost is %f" % sess.run(cost, feed_dict = {x: sim_x, y: sim_y}))
    print("Bias is {0}, estimate gives {1}".format(bias, sess.run(b)))
    print("Weight is {0}, estimate gives {1}".format(weight, sess.run(w)))



