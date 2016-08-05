## Load the data
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper('/home/mk/Github/deep_learning/mnist/data/mnist.pkl.gz')

size = [784, 30, 10]
## Model 1
import fnn
net = fnn.fnn(size)
net.train(training_data, 30, 10, 3.0, test_data=test_data)

## Model 2
import network
network = network.Network(size)
network.SGD(training_data, 30, 10, 3.0, test_data=test_data)
