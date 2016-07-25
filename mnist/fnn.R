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

    w0 = weights[[1]]
    w1 = weights[[2]]
    n = NROW(data)

    n_sample = floor(n * sampling_pct)

    while(i <= maxIter){
        if(stochastic){
            index_sample = sample(n, n_sample)
            train_data = data[index_sample, ]
            train_data_label = label[index_sample, ]
        }


        ## Forward propagation
        t1 = translation(train_data, w0)
        a1 = activation(t1)
        t2 = translation(a1, w1)
        a2 = activation(t2)

        ## print(head(a2))
        ## print(head(train_data_label_bin))

        c = cost(train_data_label, a2)
        message("Cost: ", c)
        if((c - c_old) < tol)
            break

        ## Calculate the number correctly classified
        pred = apply(a2, 1, which.max) - 1
        actual_label = apply(train_data_label, 1, which.max) - 1
        message("Percentage classified correctly: ",
                round(sum(actual_label == pred)/n_sample * 100, 4), "%")

        ## c_old = c
        i = i + 1

        ## Back propagation
        ##
        dc_da2 = cost_delta(train_data_label, a2)
        da2_dt2 = activation_delta(t2)
        dt2_dw = t(translation_delta_weight(a1))
        dt2_da1 = translation_delta_x(w1)
        da1_dt1 = activation_delta(t1)
        dt1_dw = t(translation_delta_weight(train_data))

        w0 = w0 -
            gamma * t(t(((dc_da2 * da2_dt2) %*% dt2_da1) * da1_dt1) %*% dt1_dw)
        w1 = w1 - gamma * t(t(dc_da2 * da2_dt2) %*% dt2_dw)

    }
}
