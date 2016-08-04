# import mnist_loader
# import network
# import numpy as np

# training_data, validation_data, test_data = mnist_loader.load_data_wrapper('/home/mk/Github/deep_learning/mnist/data/mnist.pkl.gz')

## Try to convert the above data to Tensorflow friendly format

## Import the data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

## ---------------------------------------------------------------------
## Initialise and create the graph
## ---------------------------------------------------------------------

## Initialise input and output
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

## Specify the model
## multi-nomial regression
# w = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# hidden1 = tf.matmul(x, w) + b
# y_hat = tf.nn.softmax(hidden1)

## two layer multi-nomial regression
w1 = tf.Variable(tf.zeros([784, 15]))
b1 = tf.Variable(tf.zeros([15]))
hidden1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
w2 = tf.Variable(tf.zeros([15, 10]))
b2 = tf.Variable(tf.zeros([10]))
hidden2 = tf.matmul(hidden1, w2) + b2
y_hat = tf.nn.softmax(hidden2)

## Specify the cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat),
                                              reduction_indices=[1]))

## Specify the training algorithm
train_step = tf.train.GradientDescentOptimizer(0.8).minimize(cross_entropy)


## Speficy the comparison
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))

## Specify the evaluation
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Initialise variables. This means x, w, b and y_hat
init = tf.initialize_all_variables()


## ---------------------------------------------------------------------
## Create the session and train the model
## ---------------------------------------------------------------------

## Start the session
sess = tf.Session()
sess.run(init)

## NOTE (Michael): Create a grid here to search for batch_size, iteration and
##                 gamma.

## Start the training
for i in range(10000):
  batch_size = 50
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
  if(i % 500 == 0):
    print("batch " + str(i))
    print(sess.run(accuracy,
                   feed_dict={x: mnist.train.images, y: mnist.train.labels}))

## ---------------------------------------------------------------------
## Check predictions
## ---------------------------------------------------------------------


## Make prediction on the test and validation dataset
print(sess.run(accuracy,
               feed_dict={x: mnist.test.images, y: mnist.test.labels}))

print(sess.run(accuracy,
               feed_dict={x: mnist.validation.images, y: mnist.validation.labels}))
