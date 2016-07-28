if(!"train_data" %in% ls()){
    ## Load the data if not already loaded.
    source("mnist_loader.R")

    ## Subset the data
    n = 1000
    train_data = train_data[1:n, ]
    train_data_label = train_data_label[1:n]

    ## Include intercept
    train_data = cbind(1, train_data)
    dimnames(train_data) = NULL

    test_data = cbind(1, test_data)
    dimnames(test_data) = NULL

    ## Convert the binary to 10 classes
    train_data_label_bin =
        matrix(0,
               nc = length(unique(train_data_label)),
               nr = length(train_data_label))

    ind = matrix(c(1:length(train_data_label),
                   train_data_label + 1), nc = 2)
    train_data_label_bin[ind] = 1

}
source("fnn.R")

## Initialisation
size = c(785, 50, 30, 15, 10)
gamma = 1e-3
maxIter = 1000
tol = 1e-10

model = fnn(data = train_data,
            label = train_data_label_bin,
            size = c(785, 30, 10),
            costFUN = cross_entropy,
            activationFUN = sigmoid,
            costDerivFUN = cross_entropy_delta,
            activationDerivFUN = sigmoid_delta,
            maxIter = maxIter,
            tol = tol,
            gamma = gamma,
            sampling_pct = 0.1)

predicted = predict(test_data, model)

