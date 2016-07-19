## Need to execute the mnist_data_converter.py
mnist = read.csv(file = "mnist.csv")

train_data = as.matrix(mnist[mnist$set == "train", 2:785])
train_data_label = as.matrix(mnist[mnist$set == "train", 786])

## Structure the following scripts

## 1. Data loader
##
## 2. Network builder
##
## 3. Executor
