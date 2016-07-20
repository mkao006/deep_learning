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


