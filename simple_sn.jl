module SNetwork

using Distributions
export Network,
    update!,
    update_weights!,
    run!,
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
    train_k_fold

import Statistics.mean,
    Random.randperm

mutable struct Network
    spikes::Dict{Int, Array{Bool,2}}
    traces::Dict{Int, Array{Bool,2}}
    potentials::Dict{Int, Array{<:Real,2}}
    connections::Dict{Tuple{Int,Int}, Array{<:Real,2}}

    net_shape::Array{Int,1}
    input_rates::Array{<:Real,1}
    teacher_rates::Array{<:Real,1}
    n_layers::Int

    thresholds::Array{<:Real,1}
    reset::Real
    memory::Int
    step::Int
    steps_per_second::Int
    max_fr::Int
    learn_rate::Real
    train_step::Int
end

function Network(net_shape::Array{<:Int,1}, mean::Real, std::Real, memory::Int)
    n_layers = length(net_shape)
    if (n_layers < 2)
        error("Must have at least 2 layers to make a network.")
    end

    #declare the space for the network's variables/objects
    spikes = Dict{Int, Array{Bool,2}}()
    traces = Dict{Int, Array{Bool,2}}()
    potentials = Dict{Int, Array{Float64,2}}()
    connections = Dict{Tuple{Int,Int}, Array{Float64,2}}()

    rand_dist = Normal(mean, std)

    #setup the input layer
    n_inputs = net_shape[1]
    input_rates = zeros(Float64, n_inputs)
    potentials[1] = -1 .* ones(Float64, n_inputs, 1)
    spikes[1] = falses(n_inputs, 1)
    traces[1] = falses(n_inputs, memory)

    #setup the hidden and output layers
    for dst in 2:n_layers
        src = dst - 1
        layer_size = net_shape[dst]
        potentials[dst] = zeros(Float64, layer_size, 1)
        spikes[dst] = falses(layer_size, 1)
        traces[dst] = falses(layer_size, memory)
        connections[(src, dst)] = rand(rand_dist, net_shape[src], net_shape[dst])
    end

    #set up the teacher signal
    n_outputs = net_shape[end]
    teacher_rates = zeros(Float64, n_outputs)
    spikes[n_layers + 1] = falses(n_outputs, 1)

    return Network(spikes, traces, potentials, connections,
        net_shape, input_rates, teacher_rates, n_layers,
        ones(n_layers), 0.0, memory, 0, 1000, 250, 0.0005, 0)
end

function update!(net::Network)
    spikes = net.spikes
    traces = net.traces
    potentials = net.potentials
    connections = net.connections

    memtime() = mod1(net.step, net.memory)
    #calculate random spiking at the inputs
    spikes[1][:,1] = rand(net.net_shape[1]) .< net.input_rates
    traces[1][:,memtime()] .= spikes[1][:,1]

    #calculate spiking in the hidden & output layers
    for i in 2:net.n_layers
        src = i - 1
        dst = i

        #add the charge from spikes in the last layer
        currents = transpose(transpose(spikes[src]) * connections[(src, dst)])
        potentials[dst] .+= currents
        #did this cause spikes?
        spikes[dst] .= potentials[dst] .>= net.thresholds[dst]
        #update the synaptic trace
        traces[dst][:,memtime()] .= spikes[dst][:,1]
        #reset potential of spiking neurons to zero
        potentials[dst][spikes[dst]] .= net.reset
    end

    #calculate the teacher signal for error propagation
    spikes[net.n_layers + 1][:,1] = rand(net.net_shape[end]) .< net.teacher_rates

    net.step += 1
end

function set_input(net::Network, input_rates::Array{<:Real,1})
    if size(input_rates,1) != net.net_shape[1]
        error("Shape of input rates must match network input layer.")
    end
    #scale input weights with respect to the maximum firing rate
    net.input_rates = (input_rates * net.max_fr / net.steps_per_second)
    reset!(net)
end

function set_teacher(net::Network, teacher_rates::Array{<:Real,1})
    if size(teacher_rates,1) != net.net_shape[end]
        error("Shape of teacher rates must match network output layer.")
    end
    #scale teacher spike rates with respect to the maximum firing rate
    net.teacher_rates .= (teacher_rates .* (net.max_fr / net.steps_per_second))
    reset!(net)
end

function update_weights!(net::Network)
    if length(net.net_shape) != 3
        error("Algorithm currently only works for networks with 1 hidden layer (Input, Hidden, & Output)")
    end

    spikes = net.spikes
    traces = net.traces
    potentials = net.potentials
    connections = net.connections
    lr = net.learn_rate

    #only trigger if there's a spike in the teacher signal
    if sum(spikes[4]) == 0
        return (zeros(net.net_shape[3]), zeros(net.net_shape[2]))
    end

    spikes_in_traces(x::Int) = sum(traces[x], dims=2)
    active_in_traces(x::Int) = (spikes_in_traces(x) .> 0)

    error_output = spikes[4] .- active_in_traces(3)
    error_hidden = connections[(2,3)] * error_output .* active_in_traces(2)

    deltas_23 = spikes_in_traces(2) * error_output' .* lr
    deltas_12 = spikes_in_traces(1) * error_hidden' .* lr

    connections[(2,3)] .+= deltas_23
    connections[(1,2)] .+= deltas_12

    net.train_step += 1

    return (error_output, error_hidden)
end

function run!(net::Network, time::Int)
    spikes = net.spikes
    potentials = net.potentials
    connections = net.connections
    n_layers = net.n_layers

    history = Dict{Int, Array{<:Real,2}}()
    output = Dict{Int, Array{Bool,2}}()

    for i in 1:n_layers
        history[i] = zeros(time, net.net_shape[i])
        output[i] = falses(time, net.net_shape[i])
    end

    for i in 1:time
        update!(net)
        for j in 1:n_layers
            history[j][i,:] = potentials[j]
            output[j][i,:] = spikes[j]
        end
    end

    return (history, output)
end

function train!(net::Network, time::Int)
    spikes = net.spikes
    potentials = net.potentials
    connections = net.connections
    n_layers = net.n_layers

    output_error = zeros(Float64, time, net.net_shape[end])
    hidden_error = zeros(Float64, time, net.net_shape[end-1])

    for i in 1:time
        update!(net)
        (output_error[i,:], hidden_error[i,:]) = update_weights!(net)
    end

    return (output_error, hidden_error)
end

function reset!(net::Network)
    potentials = net.potentials
    n_layers = net.n_layers

    #skip the
    for i in 2:n_layers
        potentials[i] .= 0
    end
end

function epoch(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int)
    (n_x, s_x) = size(x)
    (n_y, s_y) = size(y)

    check_dims(x,y)

    time_per_example = floor(Int, time/n_x)
    err = zeros(Float64, n_x)
    shuffle_inds = randperm(n_x)

    for i in 1:n_x
        #set the input firing rate to the example
        set_input(net, x[shuffle_inds[i],:])
        #set the teacher rate to the example
        set_teacher(net, y[shuffle_inds[i],:])
        #let the network learn with the update rule
        err[i] = mapreduce(sum, +, train!(net, time)) / time
    end

    return err
end

function full_test(net::Network, x::Array{<:Real,2}, time::Int)
    (n_x, s_x) = size(x)

    if s_x != net.net_shape[1]
        error("Second dimension of examples must match input dimension of network")
    end

    time_per_example = floor(Int, time/n_x)
    spikes = Array{Any,1}(undef,0)
    voltages = Array{Any,1}(undef,0)

    for i in 1:n_x
        #see how the network responds to the example stimulus
        set_input(net, x[i,:])
        (v,s) = run!(net, time_per_example)
        #get the output firing rate
        append!(spikes, s)
        append!(voltages, v)
    end

    return (spikes, voltages)
end

function test(net::Network, x::Array{<:Real,2}, time::Int)
    (n_x, s_x) = size(x)

    if s_x != net.net_shape[1]
        error("Second dimension of examples must match input dimension of network")
    end

    rates = zeros(n_x, net.net_shape[end])

    for i in 1:n_x
        #see how the network responds to the example stimulus
        set_input(net, x[i,:])
        (v,s) = run!(net, time)
        #get the output firing rate
        rates[i,:] = vec(sum(s[net.n_layers], dims=1) ./ time)
    end

    return rates
end

function check_dims(x::Array{<:Real,2}, y::Array{<:Real,2})
    if s_x != net.net_shape[1]
        error("Second dimension of examples must match input dimension of network")
    elseif n_x != n_y
        error("Each training example (x) must have a target firing rate (y)")
    elseif s_y != net.net_shape[end]
        error("Second dimension of target firing rate must match output dimension of network")
    end
end

MSE(x,y) = mean((y - x).^2)

function accuracy(x::Array{<:Real,2}, y::Array{<:Real,2})
    return mean(getindex.(findmax(x, dims=2)[2],2) .== getindex.(findmax(y, dims=2)[2],2))
end

function train_loop(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int, cycles::Int)
    mse = zeros(cycles)

    for i in 1:cycles
        err = epoch(net, x, y, time)
        mse[i] = mean(err .^ 2)
    end

    return mse
end

function train_batch(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int, cycles::Int, batch_size::Int)
    check_dims(x,y)

    n_x = size(x,1)
    if (mod(n_x, cycles) != 0)
        error("Batch size must evenly divide number of samples.")


    order = randperm(n_x)
    iteration = 1
    max_iters = n_x / batch_size
    errs = zeros(Float64, cycles)

    for i in 1:cycles
        start_i = mod1(batch_size * (i - 1) + 1, max_iters)
        stop_i = mod1(batch_size * i, max_iters)
        errs[i] = epoch(net, x[order[start_i:stop_i]], y[order[start_i:stop_i]], time)
        iteration += 1

        if iteration == max_iters
            order = randperm(n_x)
            iteration += 1
        end
    end

    return errs
end

function accuracy(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int)
    (n_x, s_x) = size(x)
    (n_y, s_y) = size(y)

    check_dims(x,y)

    rates = test(net, x, time)
    return accuracy(rates, y)

end

function train_k_fold(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int, cycles::Int, k::Int)
    check_dims(x,y)

    n_x = size(x,1)
    if (mod(n_x, k) != 0)
        error("Fold must evenly divide number of samples.")

    order = randperm(n_x)
    fold_size = n_x / k
    errs = zeros(Float64, cycles)
    acc = zeros(Float64, cycles)

    for i in 1:cycles
        fold = mod1(i, k)
        #get the indices of the fold which are being used for cross-validation
        test_ind_start = fold_size * (fold - 1) + 1
        test_ind_stop = fold_size * fold
        #get the indices of the other available samples
        train_inds = mod1.(collect(test_ind_stop + 1, test_ind_stop + (k-1) - 1), n_x)

        err[i] = epoch(net, x[train_inds], y[train_inds], time)
        acc[i] = accuracy(net, x[test_ind_start:test_ind_stop], x[test_ind_start:test_ind_stop], time*5)
    end

    return (err, acc)
end


function get_weights(net::Network)
    return deepcopy(net.connections)
end

function set_weights(net::Network, new_weights::Dict{Tuple{Int,Int}, Array{Float64,2}})
    connections = net.connections
    n_layers = net.n_layers
    net_shape = net.net_shape

    for key in keys(new_weights)
        if !(key in keys(connections))
            error("Layer (", key, ") does not exist in target network.")
        end
        if size(new_weights[key]) != size(connections[key])
            error("Size of weights for layer (", key, ") does not match target network.")
        end
    end

    for key in keys(new_weights)
        net.connections[key] = new_weights[key]
    end
end

function to_categorical(x::Array{<:Int,1})
    n_x = size(x,1)
    n_y = length(unique(x))
    categories = zeros(Float64, n_x, n_y)
    for i in 1:n_x
        categories[i,x[i]+1] = 1.0
    end
    return categories
end


end
