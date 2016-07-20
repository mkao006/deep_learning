source("mnist_loader.R")
source("fnn.R")

## Subset the data
train_data = train_data[1:100, ]
train_data_label = train_data_label[1:100]

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
n = nrow(train_data)
input_size = ncol(train_data)
hidden_size = 30
output_size = 10
gamma = 1e-3
maxIter = 1000
tol = 1e-10

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

i = 1
c_old = -Inf
while(i <= maxIter){
    ## Forward propagation
    t1 = translation(train_data, w0)
    a1 = activation(t1)
    t2 = translation(a1, w1)
    a2 = activation(t2)

    ## print(head(a2))
    ## print(head(train_data_label_bin))

    c = cost(train_data_label_bin, a2)
    message("Cost: ", c)
    ## if((c - c_old) < tol)
    ##     break

    ## Calculate the number correctly classified
    pred = apply(a2, 1, which.max) - 1
    print(sum(train_data_label == pred)/n)

    ## c_old = c
    i = i + 1

    ## Back propagation
    ##
    dc_da2 = cost_delta(train_data_label_bin, a2)
    da2_dt2 = activation_delta(t2)
    dt2_dw = t(translation_delta_weight(a1))
    dt2_da1 = translation_delta_x(w1)
    da1_dt1 = activation_delta(t1)
    dt1_dw = t(translation_delta_weight(train_data))

    w0 = w0 - gamma * t(t(((dc_da2 * da2_dt2) %*% dt2_da1) * da1_dt1) %*% dt1_dw)
    w1 = w1 - gamma * t(t(dc_da2 * da2_dt2) %*% dt2_dw)

}
