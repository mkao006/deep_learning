import numpy as np

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


def transformation(x, w, b):
    return np.dot(x, w) + b

def transformation_delta_weights(x):
    return x.T

def transformation_delta_biases(x):
    return np.ones(x.shape[0])

def transformation_delta_activation(w):
    w.T





class fnn(object):

    def __init__(self, size):
        self.size = size
        self.num_layer = len(size)
        self.biases = [np.random.randn(nodes, 1) for nodes in self.size[1:]]
        self.weights = [np.random.randn(output, input)
                        for input, output in zip(size[:-1], size[1:])]
        self.activation = []
        self.transformation = []

    def feedforward(self, x):
        """Performs the feedforward, the transformation and the activation layers
        are updated.

        """
        ## Initialise, although we can assign x to activation and then change
        ## the index
        self.transformation[0] = np.dot(x, self.weight[0]) + self.biases[0]
        self.activation[0] = sigmoid(self.transformation[0])

        for layer in range(1, len(self.size) - 1):
            self.transformation[layer] = np.dot(self.activation[layer - 1],
                                                self.weight[layer]) + \
                                                self.biases[layer]
            self.activation[layer] = sigmoid(self.transformation[layer])


    def backpropagation(self, x, y):
        """Take the activation and trasformation layer, and updates the biases and
        weights

        """
        ## Initialise the nabla (gradient)
        nabla_weights = [np.zeroes(w.shape) for w in self.weights]
        nabla_biases = [np.zeros(b.shape) for b in self.biases]

        


    def train_mini_batch(self):
        """Input is the shuffled data, and calls the feedforward and backpropagation
        method to update the biase, weights, activation and transformation.

        """

        ## Do feedforward

        ## Do back propagation

        ## Update weights

    def train(self, training_data, epochs, mini_batch_size, gamma,
              test_data = None):

        for epoch in xrange(epochs):
            ## Shuffle the data

            ## Create the mini batches

            ## train with each mini batch using back propagation

            ## Update the weights and biases








size = [784, 30, 10]
test = fnn(size)
