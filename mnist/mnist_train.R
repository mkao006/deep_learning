source("mnist_loader.R")
source("fnn.R")

## Subset the data
n = 10000
train_data = train_data[1:n, ]
train_data_label = train_data_label[1:n]

## Include intercept
train_data = cbind(1, train_data)
dimnames(train_data) = NULL

## Convert the binary to 10 classes
train_data_label_bin =
    matrix(0,
           nc = length(unique(train_data_label)),
           nr = length(train_data_label))

ind = matrix(c(1:length(train_data_label),
                        train_data_label + 1), nc = 2)
train_data_label_bin[ind] = 1

## Initialisation
input_size = ncol(train_data)
hidden_size = 30
output_size = 10
gamma = 1e-3
maxIter = 1000
tol = 1e-10
w0 = initialise_weight(input_size, hidden_size)
w1 = initialise_weight(hidden_size, output_size)

model = train(data = train_data, label = train_data_label_bin,
              weights = list(w0, w1), maxIter = 10000, tol = 1e-03,
              gamma = 1e-3, sampling_pct = 0.1)


