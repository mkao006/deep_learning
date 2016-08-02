import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper('/home/mk/Github/deep_learning/mnist/data/mnist.pkl.gz')


net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
