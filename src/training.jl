"""
Return the proportion of examples which have a maximum firing rate that matches
the desired output class (basic accuracy).
"""
function accuracy(x::Array{<:Real,2}, y::Array{<:Real,2})
    return mean(getindex.(findmax(x, dims=2)[2],2) .== getindex.(findmax(y, dims=2)[2],2))
end

"""
Given a set of examples and truths, calculate the accuracy of the spiking network
with each example evaluated for (time) steps.
"""
function accuracy(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int)
    (n_x, s_x) = size(x)
    (n_y, s_y) = size(y)

    check_dims(net, x, y)

    rates = test(net, x, time)
    return accuracy(rates, y)

end

"""
Check that the dimensions of inputs and outputs are consistent to one another
and the network structure.
"""
function check_dims(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2})
    (n_x, s_x) = size(x)
    (n_y, s_y) = size(y)

    if s_x != net.net_shape[1]
        error("Second dimension of examples must match input dimension of network")
    elseif n_x != n_y
        error("Each training example (x) must have a target firing rate (y)")
    elseif s_y != net.net_shape[end]
        error("Second dimension of target firing rate must match output dimension of network")
    end
end

"""
Carry out a training epoch, training the network on the examples and truths provided
with an inference time of (time) steps on each example.
"""
function epoch(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int)
    (n_x, s_x) = size(x)
    (n_y, s_y) = size(y)

    check_dims(net, x, y)

    time_per_example = floor(Int, time/n_x)
    err = zeros(Float64, n_x)
    shuffle_inds = randperm(n_x)

    for i in 1:n_x
        #set the input firing rate to the example
        set_input(net, x[shuffle_inds[i],:])
        #set the teacher rate to the example
        set_teacher(net, y[shuffle_inds[i],:])
        #let the network learn with the update rule
        errors = train!(net, time)
        err[i] = (sum(errors) / time)^2
    end

    return err
end

MSE(x,y) = mean((y - x).^2)

"""
Helper function to convert a list of truths in integer format (x <: [0,1,2...n]), x -> (m)
to a 2-D categorical vector (x <: [0,1]), x -> (m x classes)
"""
function to_categorical(x::Array{<:Int,1})
    n_x = size(x,1)
    n_y = length(unique(x))
    categories = zeros(Float64, n_x, n_y)
    for i in 1:n_x
        categories[i,x[i]+1] = 1.0
    end
    return categories
end

"""
Dividing the inputs up into batches, carry out (cycles) training steps where each step is a batch.
Currently updates are not batch-averaged. Inferences are run for (time) steps.
"""
function train_batch(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int, cycles::Int, batch_size::Int)
    check_dims(net, x, y)

    n_x = size(x,1)
    if (mod(n_x, cycles) != 0)
        error("Batch size must evenly divide number of samples.")
    end

    order = randperm(n_x)
    iteration = 1
    max_iters = Int(n_x / batch_size)
    errs = zeros(Float64, cycles)

    for i in 1:cycles
        start_i = mod1(batch_size * (i - 1) + 1, max_iters)
        stop_i = mod1(batch_size * i, max_iters)
        errs[i] = mean(epoch(net, x[order[start_i:stop_i],:], y[order[start_i:stop_i],:], time))
        iteration += 1

        if iteration == max_iters
            order = randperm(n_x)
            iteration += 1
        end
    end

    return errs
end

"""
Given a set of examples and truths, repeatedly train the network on this complete set for (cycles)
epochs. Inferences are run for (time) steps.
"""
function train_loop(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int, cycles::Int)
    mse = zeros(cycles)

    for i in 1:cycles
        err = epoch(net, x, y, time)
        mse[i] = mean(err .^ 2)
    end

    return mse
end

"""
Given a set of examples and truths, divide them into (k) sets. Train on (k-1) examples
and validate on the remaining set for (cycles) epochs. Inferences run for (time) steps.
"""
function train_k_fold(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int, cycles::Int, k::Int)
    check_dims(net, x, y)

    n_x = size(x,1)
    if (mod(n_x, k) != 0)
        error("Fold must evenly divide number of samples.")
    end

    order = randperm(n_x)
    fold_size = Int(n_x / k)
    err = zeros(Float64, cycles)
    acc = zeros(Float64, cycles)

    for i in 1:cycles
        fold = mod1(i, k)
        #get the indices of the fold which are being used for cross-validation
        test_ind_start = fold_size * (fold - 1) + 1
        test_ind_stop = fold_size * fold
        #get the indices of the other available samples
        train_inds = mod1.(collect(test_ind_stop + 1:test_ind_stop + (k-1) - 1), n_x)

        err[i] = sum(epoch(net, x[train_inds, :], y[train_inds, :], time))
        acc[i] = mean(accuracy(net, x[test_ind_start:test_ind_stop, :], y[test_ind_start:test_ind_stop, :], time*5))
    end

    return (err, acc)
end
