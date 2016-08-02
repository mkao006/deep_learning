source("fnn.R")
if(!"train_data" %in% ls()){
    ## Load the data if not already loaded.
    source("mnist_loader.R")

    ## Subset the data
    n = 10000
    train_data = train_data[1:n, ]
    train_data_label = train_data_label[1:n]

    ## Convert the binary to 10 classes
    train_data_label_bin =
        class2bin(train_data_label)
}




## Initialisation
size = c(785, 30, 10)
gamma = 1e-4
maxIter = 500

## Build the model
model = fnn(data = train_data,
            label = train_data_label_bin,
            size = size,
            costFUN = cross_entropy,
            activationFUN = sigmoid,
            costDerivFUN = cross_entropy_delta,
            activationDerivFUN = sigmoid_delta,
            maxIter = maxIter,
            gamma = gamma,
            batchSize = 20,
            samplingPct = 0.3)

## Make prediction
trainPredicted = predict(train_data, model)
trainPredictedClass = bin2class(trainPredicted)
sum(train_data_label == trainPredictedClass)/length(train_data_label)

testPredicted = predict(test_data, model)
testPredictedClass = bin2class(testPredicted)
sum(test_data_label == testPredictedClass)/length(test_data_label)

validationPredicted = predict(validation_data, model)
validationPredictedClass = bin2class(validationPredicted)
sum(validation_data_label == validationPredictedClass)/length(validation_data_label)


