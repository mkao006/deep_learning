import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


## Initialise parameters
size = [784, 30, 10]
gamma = 0.05
decay = 0.9
batch_size = 300
iter = 15001
p_keep = [0.8, 0.5]

## Initialise the placeholder for data
data = tf.placeholder(tf.float32, [None, size[0]])
label = tf.placeholder(tf.float32, [None, size[2]])
p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

## Initialise the variables
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

def init_bias(size):
    return tf.Variable(tf.zeros(size))

w_h = init_weight(size[:-1])
w_h2 = init_weight([size[1], size[1]])
w_o = init_weight(size[1:])
b_h = init_bias([size[1]])
b_h2 = init_bias([size[1]])
b_o = init_bias([size[2]])

## Create the model
def nnet(data, w_h, w_h2, w_o, b_h, b_h2, b_o, p_keep_input, p_keep_hidden):
    dropped = tf.nn.dropout(data, p_keep_input)
    h = tf.nn.relu(tf.matmul(dropped, w_h) + b_h)
    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2) + b_h2)
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    return tf.matmul(h2, w_o) + b_o

# Create the fitted value
y_hat = nnet(data, w_h, w_h2, w_o, b_h, b_h2, b_o, p_keep_input, p_keep_hidden)

## Construct the cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, label))

## construct an optimizer
train_alg = tf.train.RMSPropOptimizer(gamma, decay).minimize(cost)

## Speficy the comparison
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(label, 1))

## Specify the evaluation
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Initialise variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    ## Start the iteration
    for i in xrange(1, iter):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_alg, feed_dict={data: batch_xs,
                                       label: batch_ys,
                                       p_keep_input: p_keep[0],
                                       p_keep_hidden: p_keep[1]})
        if(i % 1000 == 0):
            print("batch " + str(i))
            print(sess.run(accuracy,
                           feed_dict = {data: mnist.train.images,
                                        label: mnist.train.labels,
                                         p_keep_input: p_keep[0],
                                         p_keep_hidden: p_keep[1]}))
            print(sess.run(accuracy,
                           feed_dict = {data: mnist.test.images,
                                        label: mnist.test.labels,
                                         p_keep_input: 1.0,
                                         p_keep_hidden: 1.0}))
