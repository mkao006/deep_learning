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

translation = function(x, w){
    x %*% w
}

cross_entropy_delta = function(p, q){
    k = 1e-15
    sq = ifelse(q == 0, k, ifelse(q == 1, 1 - k, q))
    (p - sq)/(sq * (sq - 1))
}

sigmoid_delta = function(x){
    exp(x)/((1 + exp(x))^2)
}

translation_delta_weight= function(x){
    t(x)
}

translation_delta_x = function(w){
    t(w)
}

initialise_weights = function(size){
    layers = length(size) - 1
    weights = vector(mode = "list", layers)
    for(layer in 1:layers){
        weights[[layer]] =
            apply(X = matrix(rnorm(size[layer] * size[layer + 1]),
                             nr = size[layer],
                             nc = size[layer + 1]),
                  MARGIN = 2,
                  FUN = function(x) x/sum(x))
    }
    weights
}

fp = function(data, weights, activationFUN){

    n.layers = length(weights) + 1

    translation_layer = vector("list", n.layers)
    translation_layer[[1]] = data
    activation_layer = translation_layer


    for(layer in 1:(n.layers - 1)){
        translation_layer[[layer + 1]] =
            translation(translation_layer[[layer]], weights[[layer]])
        activation_layer[[layer + 1]] =
            activationFUN(translation_layer[[layer + 1]])
    }
    list(translation_layer = translation_layer,
         activation_layer = activation_layer)
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
               sampling_pct = 0.3,
               gamma = 1e-3){

    ## Check
    if(ncol(data) != size[1])
        stop("Incorrect input size")
    if(ncol(label) != size[length(size)])
        stop("Incorrect output size")

    ## Initialisation
    n.sample= size[1]
    n.layers = length(size)
    weights = initialise_weights(size)
    i = 1

    dt_dw = vector("list", n.layers - 1)
    dt_da = vector("list", n.layers - 2)
    da_dt = vector("list", n.layers - 1)

    ## Start stochastic gradient descent
    while(i <= maxIter){
        if(stochastic){
            n_sample = floor(n.sample * sampling_pct)
            index_sample = sample(n.sample, n_sample)
            train_data = data[index_sample, ]
            train_data_label = label[index_sample, ]
        } else {
            n_sample = n.sample
            train_data = data
            train_data_label = label
        }

        ## Forward propagation
        forward = fp(data = train_data,
                     weights = weights,
                     activationFUN = activationFUN)

        ## Compute the cost
        cost = costFUN(train_data_label, forward$activation_layer[[n.layers]])
        message("Cost: ", cost)

        ## TODO (Michael): Need a way to identify the convergence of the
        ##                 stochastic gradient descent.

        ## Calculate the number correctly classified
        pred = apply(forward$activation_layer[[n.layers]], 1, which.max) - 1
        actual_label = apply(train_data_label, 1, which.max) - 1
        message("Percentage classified correctly: ",
                round(sum(actual_label == pred)/n_sample * 100, 4), "%")

        ## Increment i
        i = i + 1

        ## Back propagation
        w_tmp = weights

        dc_da =
            costDerivFUN(train_data_label, forward$activation_layer[[n.layers]])
        da_dt[[n.layers - 1]] =
            activationDerivFUN(forward$translation_layer[[n.layers]])
        dc_dt = dc_da * da_dt[[n.layers - 1]]
        dt_dw[[n.layers - 1]] =
            t(translation_delta_weight(forward$activation_layer[[n.layers - 1]]))
        w_tmp[[n.layers - 1]] =
            w_tmp[[n.layers - 1]] - gamma * t(t(dc_dt) %*% dt_dw[[n.layers - 1]])

        for(layer in (n.layers - 2):1){
            ## Calculate individual derivatives
            da_dt[[layer]] =
                activationDerivFUN(forward$translation_layer[[layer + 1]])
            dt_dw[[layer]] =
                t(translation_delta_weight(forward$activation_layer[[layer]]))
            dt_da[[layer]] =
                translation_delta_x(weights[[layer + 1]])

            ## Update derivative of cost to each layer
            dc_dt = (dc_dt %*% dt_da[[layer]]) * da_dt[[layer]]

            ## Update the weights
            w_tmp[[layer]] =
                w_tmp[[layer]] - gamma * t(t(dc_dt) %*% dt_dw[[layer]])
        }
        weights = w_tmp
    }

    ## Return model
    model = list(data = data,
                 label = label,
                 size = size,
                 costFUN = costFUN,
                 activationFUN = activationFUN,
                 weights = weights)
    class(model) = "fnn"
    model
}

predict = function(data, model){
    n.layers = length(model$size)
    forward =
        with(model,
             fp(data, model$weights, activationFUN))
    forward$activation_layer[n.layers]
}
