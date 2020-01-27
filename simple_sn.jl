module SNetwork

using Distributions
export Network,
    update!,
    update_weights!,
    run!,
    full_run!,
    train!,
    reset!,
    epoch,
    test,
    full_test,
    set_input,
    set_teacher,
    train_loop,
    MSE,
    accuracy,
    get_weights,
    set_weights,
    train_batch,
    train_k_fold,
    to_categorical

import Statistics.mean,
    Random.randperm

include("src/network.jl")
include("src/training.jl")

end
