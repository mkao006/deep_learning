import cPickle
import gzip
import pandas
import feather

def load_data(file_name):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open(file_name, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


mnist = load_data("mnist.pkl.gz")



train_data = pandas.DataFrame(mnist[0][0])
train_data['label'] = mnist[0][1]
train_data['set'] = "train"
validation_data = pandas.DataFrame(mnist[1][0])
validation_data['label'] = mnist[1][1]
validation_data['set'] = "validation"
test_data = pandas.DataFrame(mnist[2][0])
test_data['label'] = mnist[1][1]
test_data['set'] = "test"

train_data = train_data.append(validation_data)
train_data = train_data.append(test_data)


# path = 'mnist.feather'
# feather.write_dataframe(train_data, path)
train_data.to_csv("mnist.csv")
