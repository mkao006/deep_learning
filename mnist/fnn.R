cost = function(p, q){
    ## NOTE (Michael): This is to avoid the cross-entropy to be undefined when q
    ##                 is 1 or 0. However, there should be a better way of
    ##                 dealing with this.
    k = 1e-10
    sq = ifelse(q == 0, k, ifelse(q == 1, 1 - k, q))
    s = p * log(sq) + (1 - p) * log(1 - sq)
    -sum(s) * (1/NROW(q))
}

activation = function(x){
    1/(1 + exp(-x))
}

translation = function(x, w){
    x %*% w
}

cost_delta = function(p, q){
    k = 1e-15
    sq = ifelse(q == 0, k, ifelse(q == 1, 1 - k, q))
    (p - sq)/(sq * (sq - 1))
}

activation_delta = function(x){
    exp(x)/((1 + exp(x))^2)
}

translation_delta_weight= function(x){
    t(x)
}

translation_delta_x = function(w){
    t(w)
}

initialise_weight = function(input_size, output_size){
    apply(X = matrix(rnorm(input_size * output_size),
                     nr = input_size,
                     nc = output_size),
          MARGIN = 2,
          FUN = function(x) x/sum(x))
}

train = function(data,
                 label,
                 weights,
                 maxIter,
                 tol,
                 stochastic = TRUE,
                 sampling_pct = 0.3,
                 gamma = 1e-3){
    i = 1
    c_old = -Inf



    n = NROW(data)
    size = c(ncol(data), ncol(weights[[1]]), ncol(weights[[2]]))

    while(i <= maxIter){
        if(stochastic){
            n_sample = floor(n * sampling_pct)
            index_sample = sample(n, n_sample)
            train_data = data[index_sample, ]
            train_data_label = label[index_sample, ]
        } else {
            n_sample = n
            train_data = data
            train_data_label = label
        }

        ## Initialise t and a, a has same dimension as t
        t = list(train_data,
                 matrix(0, nc = size[2], nr = size[1]),
                 matrix(0, nc = size[3], nr = size[2]))
        a = t

        dc_da = matrix(0, nr = size[1], nc = size[3])
        da_dt = list(matrix(0, nr = size[1], nc = size[2]),
                     matrix(0, nr = size[1], nc = size[3]))
        dt_dw = list(matrix(0, nr = size[1], nc = size[2]),
                     matrix(0, nr = size[1], nc = size[3]))
        dt_da = list(matrix(0, nr = size[2], size[3]),
                     1)


        ## Forward propagation
        for(layer in 1:length(weights)){
            t[[layer + 1]] = translation(t[[layer]], weights[[layer]])
            a[[layer + 1]] = activation(t[[layer + 1]])
        }

        c = cost(train_data_label, a[[3]])
        message("Cost: ", c)
        if((c - c_old) < tol)
            break

        ## Calculate the number correctly classified
        pred = apply(a[[3]], 1, which.max) - 1
        actual_label = apply(train_data_label, 1, which.max) - 1
        message("Percentage classified correctly: ",
                round(sum(actual_label == pred)/n_sample * 100, 4), "%")

        ## c_old = c
        i = i + 1

        ## Back propagation
        ##

        dc_da = cost_delta(train_data_label, a[[length(a)]])
        for(layer in 1:length(weights)){
            da_dt[[layer]] = activation_delta(t[[layer + 1]])
            dt_dw[[layer]] = t(translation_delta_weight(a[[layer]]))
            if(layer <= (length(weights) - 1))
                dt_da[[layer]] = translation_delta_x(weights[[layer + 1]])

        }

        weights[[2]] =
            weights[[2]] - gamma *
            t(t(dc_da * da_dt[[2]]) %*% dt_dw[[2]])

        weights[[1]] =
            weights[[1]] - gamma *
            t(t((dc_da * da_dt[[2]]) %*% dt_da[[1]] * da_dt[[1]]) %*%
              dt_dw[[1]])


    }
}

predict = function(data, model){
}
