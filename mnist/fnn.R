cost = function(p, q){
}

activation = function(x){
    1/(1 + exp(x))
}

translation = function(x, w){
    x %*% w
}

cost_delta = function(p, q){
}

activation_delta = function(x){
    -exp(x)/(1 + exp(x))
}

translation_delta_weight= function(x){
    t(x)
}

translation_delta_x = function(w){
    t(w)
}


