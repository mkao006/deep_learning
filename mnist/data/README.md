# Data for MNIST

The raw data **mnist.pkl.gz** was downloaded from
[mnielson](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz).

The `mnist_data_converter.load_data` function is also from the same directory.
For more information about the data, please see the doc string of the function.

The script `mnist_data_converter` converts the data to a csv file in order to
train the data in R. In the future, we will convert this to the `.feather`
format which is friendly to both `R` and `Python` (See issue #1).