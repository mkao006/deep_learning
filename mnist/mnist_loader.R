## Need to execute the mnist_data_converter.py
library(data.table)
library(magrittr)
library(dplyr)

## Read the data
mnist =
    fread(input = "mnist.csv", stringsAsFactors = FALSE, header = TRUE)


## Remove the first column which is the row names
mnist =
    mnist %>%
    select(., -V1)

train_data =
    mnist %>%
    subset(., set == "train") %>%
    select(., matches("^[0-9]")) %>%
    as.matrix

train_data_label =
    mnist %>%
    subset(., set == "train") %>%
    select(., label) %>%
    as.matrix
