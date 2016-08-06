import numpy as np
import random

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_delta(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def cross_entropy(p, q):
    """The Cross-entropy function."""
    ## This is to avoid numerical error as the entropy is undefined when q is
    ## either 0 or 1.
    k = 1e-10
    q[q == 0] = k
    q[q == 1.0] = 1.0 - k
    cross_entropy = p * np.log(q) + (1.0 - p) * np.log(1.0 - q)
    return -np.sum(cross_entropy) * 1.0/q.size


def cross_entropy_delta(p, q):
    """The derivative of the cross entropy function"""
    k = 1e-10
    q[q == 0] = k
    q[q == 1.0] = 1.0 - k
    return (p - q)/(q * (q - 1.0))/q.size


def trans(x, w, b):
    return np.dot(x, w) + b

def trans_delta_weights(x):
    return x.T

def trans_delta_biases(x):
    return np.ones(x.shape[1])

def trans_delta_activation(w):
    return w.T





class Fnn(object):

    def __init__(self, size):
        self.size = size
        self.num_layer = len(size)
        self.biases = [np.random.randn(1, nodes) for nodes in self.size[1:]]
        self.weights = [np.random.randn(input, output)
                        for input, output in zip(size[:-1], size[1:])]
        self.activation = [None] * (self.num_layer)
        self.trans = [None] * (self.num_layer - 1)

    def feedforward(self, x, y):
        """Performs the feedforward, the transformation and the activation layers
        are updated.

        """
        ## Initialise, although we can assign x to activation and then change
        ## the index
        self.activation[0] = x

        for layer in range(0, self.num_layer - 1):
            ## Perform transformation
            self.trans[layer] = np.dot(self.activation[layer],
                                       self.weights[layer]) + \
                                       self.biases[layer]
            ## Perform activation
            self.activation[layer + 1] = sigmoid(self.trans[layer])

        cost = cross_entropy(y, self.activation[-1])
        if(np.isnan(cost)):
            raise Exception("Cost is nan")
        print(cost)

    def backpropagation(self, x, y, gamma):
        """Take the activation and trasformation layer, and updates the biases and
        weights

        """
        ## NOTE (Michael): There still seems to be something wrong with the back
        ##                 propagation. Initialise the nabla (gradient)
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        nabla_biases = [np.zeros(b.shape) for b in self.biases]

        ## Derivative of cost w.r.t to transformation
        delta = cross_entropy_delta(y, self.activation[-1]) * \
                sigmoid_delta(self.trans[-1])
        ## Derivative of transformation to weights
        nabla_weights[-1] = np.dot(trans_delta_weights(self.activation[-2]),
                                   delta)
        ## Derivative of trans to bias
        nabla_biases[-1] = delta
        for layer in xrange(2, self.num_layer):
            delta = np.dot(delta,
                           trans_delta_activation(self.weights[-layer + 1]))
            delta = delta * sigmoid_delta(self.trans[-layer])
            nabla_weights[-layer] = np.dot(trans_delta_weights(self.activation[-layer - 1]),
                                           delta)
            nabla_biases[-layer] = delta

        self.weights = [w - gamma * nw
                        for w, nw in zip(self.weights, nabla_weights)]
        self.biases = [b - gamma * nb
                       for b, nb in zip(self.biases, nabla_biases)]

    def train_mini_batch(self, mini_batch, gamma):
        """Input is the shuffled data, and calls the feedforward and backpropagation
        method to update the biase, weights, activation and transformation.

        """
        for sx, sy in mini_batch:
            ## Do feedforward
            self.feedforward(sx.T, sy.T)
            ## Do back propagation
            self.backpropagation(sx.T, sy.T, gamma)


    def train(self, training_data, epochs, mini_batch_size, gamma, test_data = None):
        """Train the model"""
        for epoch in xrange(epochs):
            ## Shuffle the data
            random.shuffle(training_data)
            n = len(training_data)
            ## Create the mini batches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            ## train with each mini batch using back propagation
            for mini_batch in mini_batches:
                self.train_mini_batch(mini_batch, gamma)



import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper('/home/mk/Github/deep_learning/mnist/data/mnist.pkl.gz')

size = [784, 30, 10]
## Model 1
net = Fnn(size)
net.train(training_data, 30, 10, 0.1, test_data=test_data)
