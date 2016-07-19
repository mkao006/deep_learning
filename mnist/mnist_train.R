source("mnist_loader.R")
source("fnn.R")

## Include intercept
train_data = cbind(1, train_data)

## Initialisation
input_size = ncol(train_data)
hidden_size = 30
output_size = 10
gamma = 1e-3

w0 = apply(X = matrix(rnorm(input_size * hidden_size),
                      nr = input_size,
                      nc = hidden_size),
           MARGIN = 2,
           FUN = function(x) x/sum(x))
w1 = apply(X = matrix(rnorm(hidden_size * output_size),
                      nr = hidden_size,
                      nc = output_size),
           MARGIN = 2,
           FUN = function(x) x/sum(x))

## Forward propagation
t1 = translation(train_data, w0)
a1 = activation(t1)
t2 = translation(a1, w1)
a2 = activation(t2)

c = cost(train_data_label, a2)

## Back propagation
w0 = w0 - gamma *
    translation_delta_weight(train_data) %*%
    (cost_delta(train_data_label, a2) * activation_delta(a2) %*%
    translation_delta_x(w1) * activation_delta(a1))


w1 = w1 - gamma *
    translation_delta_weight() %*%
    (cost_delta(train_data_label, a2) * activation_delta(a2))



