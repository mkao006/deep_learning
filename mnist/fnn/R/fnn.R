cross_entropy = function(p, q){
    ## NOTE (Michael): This is to avoid the cross-entropy to be undefined when q
    ##                 is 1 or 0. However, there should be a better way of
    ##                 dealing with this.
    k = 1e-10
    sq = ifelse(q == 0, k, ifelse(q == 1, 1 - k, q))
    s = p * log(sq) + (1 - p) * log(1 - sq)
    -sum(s) * (1/NROW(q))
}

sigmoid = function(x){
    1/(1 + exp(-x))
}

translation = function(x, w, b){
    t(t(x %*% w) + c(b))
}

cross_entropy_delta = function(p, q){
    k = 1e-15
    sq = ifelse(q == 0, k, ifelse(q == 1, 1 - k, q))
    (p - sq)/(sq * (sq - 1))/(length(q))
}

sigmoid_delta = function(x){
    delta = exp(-x)/((1 + exp(-x))^2)
    ## NOTE (Michael): This is to avoid numerical error
    delta[is.nan(delta)] = 0
    delta
}

translation_delta_weight= function(x){
    t(x)
}

translation_delta_bias = function(x){
    matrix(1, nr = 1, nc = nrow(x))
}

translation_delta_x = function(w){
    t(w)
}


initialise_weights = function(size){
    layers = length(size) - 1
    weights = vector(mode = "list", layers)
    for(layer in 1:layers){
        ## weights[[layer]] =
        ##     apply(X = matrix(rnorm(size[layer] * size[layer + 1]),
        ##                      nr = size[layer],
        ##                      nc = size[layer + 1]),
        ##           MARGIN = 2,
        ##           FUN = function(x) x/sum(x))
        weights[[layer]] =
            matrix(rnorm(size[layer] * size[layer + 1]),
                   nr = size[layer],
                   nc = size[layer + 1])
    }
    weights
}

initialise_bias = function(size){
    layers = length(size) - 1
    bias = vector(mode = "list", layers)
    for(layer in 1:layers){
        bias[[layer]] = matrix(rnorm(size[layer + 1]),
                               nc = 1, nr = size[layer + 1])
    }
    bias
}



fp = function(data, weights, bias, activationFUN){

    n.layers = length(weights) + 1

    translation_layer = vector("list", n.layers)
    translation_layer[[1]] = data
    activation_layer = translation_layer


    for(layer in 1:(n.layers - 1)){
        translation_layer[[layer + 1]] =
            translation(translation_layer[[layer]], weights[[layer]], bias[[layer]])
        activation_layer[[layer + 1]] =
            activationFUN(translation_layer[[layer + 1]])
    }
    list(translation_layer = translation_layer,
         activation_layer = activation_layer)
}

bp = function(label, weights, bias, fp, costDerivFUN, activationDerivFUN, gamma){
    n.layers = length(weights) + 1
    w_tmp = weights
    b_tmp = bias
    dt_dw = vector("list", n.layers - 1)
    dt_db = vector("list", n.layers - 1)
    dt_da = vector("list", n.layers - 2)
    da_dt = vector("list", n.layers - 1)

    dc_da =
        costDerivFUN(label, fp$activation_layer[[n.layers]])
    da_dt[[n.layers - 1]] =
        activationDerivFUN(fp$translation_layer[[n.layers]])
    dc_dt = dc_da * da_dt[[n.layers - 1]]
    dt_dw[[n.layers - 1]] =
        t(translation_delta_weight(fp$activation_layer[[n.layers - 1]]))
    dt_db[[n.layers - 1]] =
        t(translation_delta_bias(fp$activation_layer[[n.layers - 1]]))
    w_tmp[[n.layers - 1]] =
        w_tmp[[n.layers - 1]] - gamma * t(t(dc_dt) %*% dt_dw[[n.layers - 1]])
    b_tmp[[n.layers - 1]] =
        b_tmp[[n.layers - 1]] - gamma * t(dc_dt) %*% dt_db[[n.layers - 1]]

    for(layer in (n.layers - 2):1){
        ## Calculate individual derivatives
        da_dt[[layer]] =
            activationDerivFUN(fp$translation_layer[[layer + 1]])
        dt_dw[[layer]] =
            t(translation_delta_weight(fp$activation_layer[[layer]]))
        dt_db[[layer]] =
            t(translation_delta_bias(fp$activation_layer[[layer]]))
        dt_da[[layer]] =
            translation_delta_x(weights[[layer + 1]])

        ## Update derivative of cost to each layer
        dc_dt = (dc_dt %*% dt_da[[layer]]) * da_dt[[layer]]

        ## Update the weights
        w_tmp[[layer]] =
            w_tmp[[layer]] - gamma * t(t(dc_dt) %*% dt_dw[[layer]])
        b_tmp[[layer]] =
            b_tmp[[layer]] - gamma * t(dc_dt) %*% dt_db[[layer]]
    }
    list(weights = w_tmp, bias = b_tmp)
}

sgd = function(data,
               label,
               size,
               costFUN,
               activationFUN,
               costDerivFUN,
               activationDerivFUN,
               maxIter,
               gamma,
               batchSize = 50){

    ## Initialisation
    n_data= nrow(data)
    n.layers = length(size)
    weights = initialise_weights(size)
    bias = initialise_bias(size)
    i = 1

    ## Start stochastic gradient descent
    while(i <= maxIter){
        shuffleIndex = sample(n_data, n_data)
        shuffledData = data[shuffleIndex, ]
        shuffledLabel = label[shuffleIndex, ]
        n_sample = floor(n_data * 1/batchSize)
        for(batch in 1:batchSize){
            index_sample = 1:n_sample + ((batch - 1) * n_sample)
            train_data = shuffledData[index_sample, ]
            train_data_label = shuffledLabel[index_sample, ]

            ## Forward propagation
            forward = fp(data = train_data,
                         weights = weights,
                         bias = bias,
                         activationFUN = activationFUN)

            ## TODO (Michael): Need a way to identify the convergence of the
            ##                 stochastic gradient descent.

            ## Back propagation
            back = bp(label = train_data_label,
                      weights = weights,
                      bias = bias,
                      fp = forward,
                      costDerivFUN = costDerivFUN,
                      activationDerivFUN = activationDerivFUN,
                      gamma)
            weights = back$weights
            bias = back$bias

        }

        finalForward =
            fp(data = shuffledData,
               weights = weights,
               bias = bias,
               activationFUN = activationFUN)
        cost =
            costFUN(shuffledLabel, finalForward$activation_layer[[n.layers]])
        message("Cost: ", cost)

        ## Calculate the number correctly classified
        pred = apply(finalForward$activation_layer[[n.layers]], 1, which.max) - 1
        actual_label = apply(shuffledLabel, 1, which.max) - 1
        message("Percentage classified correctly: ",
                round(sum(actual_label == pred)/n_data * 100, 4), "%")

        ## Increment i
        i = i + 1

        ## ## Shrink gamma
        ## gamma = gamma * 0.98
        ## print(gamma)

    }
    list(weights = weights, bias = bias)
}

rsgd = function(data,
                label,
                size,
                costFUN,
                activationFUN,
                costDerivFUN,
                activationDerivFUN,
                maxIter,
                gamma,
                batchSize = 50,
                samplingPct = 0.3){

    ## Initialisation
    n_data= nrow(data)
    n.layers = length(size)
    weights = initialise_weights(size)
    bias = initialise_bias(size)

    i = 1
    n_sample = floor(n_data * samplingPct)
    while(i <= maxIter){

        index_sample = sample(n_data, n_sample)
        train_data = data[index_sample, ]
        train_data_label = label[index_sample, ]

        ## Forward propagation
        forward = fp(data = train_data,
                     weights = weights,
                     bias = bias,
                     activationFUN = activationFUN)
        cost =
            costFUN(train_data_label, forward$activation_layer[[n.layers]])

        message("Cost: ", cost)

        ## TODO (Michael): Need a way to identify the convergence of the
        ##                 stochastic gradient descent.

        ## Back propagation
        back = bp(label = train_data_label,
                  weights = weights,
                  bias = bias,
                  fp = forward,
                  costDerivFUN = costDerivFUN,
                  activationDerivFUN = activationDerivFUN,
                  gamma)
        bias = back$bias
        weights = back$weights

        ## Calculate the number correctly classified
        pred = apply(forward$activation_layer[[n.layers]], 1, which.max) - 1
        actual_label = apply(train_data_label, 1, which.max) - 1
        message("Percentage classified correctly: ",
                round(sum(actual_label == pred)/n_sample * 100, 4), "%")

        ## Increment i
        i = i + 1

    }
    list(weights = weights, bias = bias)
}



fnn = function(data,
               label,
               size,
               costFUN,
               activationFUN,
               costDerivFUN,
               activationDerivFUN,
               maxIter,
               tol,
               stochastic = TRUE,
               gamma = 1e-3,
               batchSize = 20,
               samplingPct){

    ## Check
    if(ncol(data) != size[1])
        stop("Incorrect input size")
    if(ncol(label) != size[length(size)])
        stop("Incorrect output size")

    ## Estimate the model
    params =
        sgd(data = data,
            label = label,
            size = size,
            costFUN = costFUN,
            activationFUN = activationFUN,
            activationDerivFUN = activationDerivFUN,
            costDerivFUN = costDerivFUN,
            maxIter = maxIter,
            gamma = gamma,
            batchSize = batchSize)

    ## params =
    ##     rsgd(data = data,
    ##          label = label,
    ##          size = size,
    ##          costFUN = costFUN,
    ##          activationFUN = activationFUN,
    ##          activationDerivFUN = activationDerivFUN,
    ##          costDerivFUN = costDerivFUN,
    ##          maxIter = maxIter,
    ##          gamma = gamma,
    ##          batchSize = batchSize,
    ##          samplingPct = samplingPct)

    ## Return model
    model = list(model_data = data,
                 label = label,
                 size = size,
                 costFUN = costFUN,
                 activationFUN = activationFUN,
                 weights = params$weights,
                 bias = params$bias)
    class(model) = "fnn"
    model
}

predict = function(data, model){
    n.layers = length(model$size)
    forward =
        with(model,
             fp(data, model$weights, model$bias, activationFUN))
    forward$activation_layer[[n.layers]]
}
