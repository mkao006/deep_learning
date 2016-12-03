import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

## Initialise parameters
size = [784, 100, 10]
gamma = 0.05
batch_size = 300
iter = 15001


## Initialise the placeholder for data
data = tf.placeholder(tf.float32, [None, size[0]])
label = tf.placeholder(tf.float32, [None, size[2]])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(data, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    dropped = tf.nn.dropout(data, p_keep_input)
    h = tf.nn.relu(tf.matmul(dropped, w_h))
    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    return tf.matmul(h2, w_o)



w_h = init_weights(size[:-1])
w_h2 = init_weights([size[1], size[1]])
w_o = init_weights(size[1:])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
y_hat = model(data, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, label))
train_alg = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(y_hat, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(30000):
        batch_sx, batch_sy = mnist.train.next_batch(batch_size)
        sess.run(train_alg, feed_dict = {data: batch_sx,
                                         label: batch_sy,
                                         p_keep_input: 0.8,
                                         p_keep_hidden: 0.5})
        if(i % 3000 == 0):
            print(i, np.mean(np.argmax(mnist.test.labels, axis=1) ==
                             sess.run(predict_op, feed_dict={data: mnist.test.images,
                                                             label: mnist.test.labels,
                                                             p_keep_input: 1.0,
                                                             p_keep_hidden: 1.0})))
