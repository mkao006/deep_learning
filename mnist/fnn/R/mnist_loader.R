## Optional: This converts the data from .pkl to .csv.
##
## system("python mnist_data_converter.py")

## Load the library
library(data.table)
library(magrittr)
library(dplyr)

mnist_path = "~/Github/deep_learning/mnist/data/mnist.csv"

## Read the data
mnist =
    fread(input = mnist_path,
          stringsAsFactors = FALSE,
          header = TRUE)

## Remove the first column which is the row names
mnist =
    mnist %>%
    select(., -V1)

## Create the training data
train_data =
    mnist %>%
    subset(., set == "train") %>%
    select(., matches("^[0-9]")) %>%
    unname %>%
    as.matrix

## Create the training label
train_data_label =
    mnist %>%
    subset(., set == "train") %>%
    select(., label) %>%
    as.matrix


## Create the validation data
validation_data =
    mnist %>%
    subset(., set == "validation") %>%
    select(., matches("^[0-9]")) %>%
    as.matrix

## Create the validation label
validation_data_label =
    mnist %>%
    subset(., set == "validation") %>%
    select(., label) %>%
    unname %>%
    as.matrix


## Create the testing data
test_data =
    mnist %>%
    subset(., set == "test") %>%
    select(., matches("^[0-9]")) %>%
    unname %>%
    as.matrix

## Create the testing label
test_data_label =
    mnist %>%
    subset(., set == "test") %>%
    select(., label) %>%
    as.matrix

bin2class = function(data){
    apply(data, 1, which.max) - 1
}

class2bin = function(data){
    ## Convert the binary to 10 classes
    data_bin =
        matrix(0,
               nc = length(unique(data)),
               nr = length(data))

    ind = matrix(c(1:length(data),
                   data + 1), nc = 2)
    data_bin[ind] = 1
    data_bin
}
