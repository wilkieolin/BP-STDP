"""
Basic feed-forward spiking network setup and evaulation.
"""

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
    soft_reset::Bool
    memory::Int
    step::Int
    steps_per_second::Int
    max_fr::Int
    learn_rate::Real
    train_step::Int
end

"""
Constructor function to setup a feed-forward spiking network with ReLU-like activation.
    Takes array of the desired shape (n_inputs, n_hidden_1, ... , n_hidden_m, n_outputs)
    Mean and std for random weight initialization
    Memory is the length (number) of time steps inspected by STDP rule
"""
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
        ones(n_layers), 0.0, false, memory, 0, 1000, 250, 0.0005, 0)
end

"""
Given input firing rates, return the spikes and voltages of all layers for (time) steps.
"""
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

"""
Return a dictionary of arrays representing the weights of each layer in the network.
"""
function get_weights(net::Network)
    return deepcopy(net.connections)
end

"""
Reset the network's neuronal potentials to zero.
Necessary while changing inputs when neurons can be hyper-polarized.
"""
function reset!(net::Network)
    potentials = net.potentials
    n_layers = net.n_layers

    #skip the
    for i in 2:n_layers
        potentials[i] .= 0
    end
end

"""
Run the network for (time) steps, returning the spikes for each step
"""
function run!(net::Network, time::Int)
    spikes = net.spikes
    potentials = net.potentials
    connections = net.connections
    n_layers = net.n_layers

    output = Dict{Int, Array{Bool,2}}()

    for i in 1:n_layers
        output[i] = falses(time, net.net_shape[i])
    end

    for i in 1:time
        update!(net)
        for j in 1:n_layers
            output[j][i,:] = spikes[j]
        end
    end

    return output
end

"""
Run the network for (time) steps, returning the spikes & potentials for each step
"""
function run_full!(net::Network, time::Int)
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
"""
Given a dictionary of arrays representing the weights at each layer of the network,
assign those weights to the network.
"""
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

"""
Set the input firing rates of the network.
"""
function set_input(net::Network, input_rates::Array{<:Real,1})
    if size(input_rates,1) != net.net_shape[1]
        error("Shape of input rates must match network input layer.")
    end
    #scale input weights with respect to the maximum firing rate
    net.input_rates = (input_rates * net.max_fr / net.steps_per_second)
    reset!(net)
end

"""
Set the desired firing rates at the output layer.
"""
function set_teacher(net::Network, teacher_rates::Array{<:Real,1})
    if size(teacher_rates,1) != net.net_shape[end]
        error("Shape of teacher rates must match network output layer.")
    end
    #scale teacher spike rates with respect to the maximum firing rate
    net.teacher_rates .= (teacher_rates .* (net.max_fr / net.steps_per_second))
    reset!(net)
end

"""
Given an array of inputs, run each through the network and return the output
firing rates.
"""
function test(net::Network, x::Array{<:Real,2}, time::Int)
    (n_x, s_x) = size(x)

    if s_x != net.net_shape[1]
        error("Second dimension of examples must match input dimension of network")
    end

    rates = zeros(n_x, net.net_shape[end])

    for i in 1:n_x
        #see how the network responds to the example stimulus
        set_input(net, x[i,:])
        s = run!(net, time)
        #get the output firing rate
        rates[i,:] = vec(sum(s[net.n_layers], dims=1) ./ time)
    end

    return rates
end

"""
The basic inference and training loop for the network.
Run it for a number of steps with the current input/desired output rates,
with the learning algorithm enabled.
"""
function train!(net::Network, time::Int)
    spikes = net.spikes
    potentials = net.potentials
    connections = net.connections
    n_layers = net.n_layers

    total_errors = zeros(Float64, time, net.n_layers-1)

    for i in 1:time
        update!(net)
        err = update_weights!(net)
        total_errors[i,:] = sum.(err)
    end

    return total_errors
end

"""
Atomic update rule for the spiking network. Update potentials at each layer,
and determine which neurons have fired.
"""
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
        if USE_CUARRAYS
            gpu_spikes = CuArray(transpose(spikes[src]))
            gpu_weights = CuArray(connections[(src, dst)])
            currents = transpose(Array(gpu_spikes * gpu_weights))
        else
            currents = transpose(transpose(spikes[src]) * connections[(src, dst)])
        end
        potentials[dst] .+= currents
        #did this cause spikes?
        spikes[dst] .= potentials[dst] .>= net.thresholds[dst]
        #update the synaptic trace
        traces[dst][:,memtime()] .= spikes[dst][:,1]
        if !net.soft_reset
            #reset potential of spiking neurons to zero
            potentials[dst][spikes[dst]] .= net.reset
        else
            potentials[dst][spikes[dst]] .-= net.thresholds[i]
        end
    end

    #calculate the teacher signal for error propagation
    spikes[net.n_layers + 1][:,1] = rand(net.net_shape[end]) .< net.teacher_rates

    net.step += 1
end

"""
The atomic learning rule which will adjust weights to make the network's firing
rates match the teacher signal.
"""
function update_weights!(net::Network)
    spikes = net.spikes
    traces = net.traces
    potentials = net.potentials
    connections = net.connections
    lr = net.learn_rate

    n_layers = net.n_layers

    spikes_in_traces(x::Int) = sum(traces[x], dims=2)
    active_in_traces(x::Int) = (spikes_in_traces(x) .> 0)

    error_by_layer = Array{Array{Float64,2},1}(undef, n_layers-1)
    delta_by_layer = Array{Array{Float64,2},1}(undef, n_layers-1)

    #only trigger if there's a spike in the teacher signal
    if sum(spikes[n_layers+1]) == 0
        for i in 2:n_layers
            error_by_layer[i-1] = zeros(Float64, net.net_shape[i], 1)
        end
    else
        for i in n_layers:-1:2
            #is this the output layer?
            if i == n_layers
                #calculate its error from the teacher signal
                error_by_layer[i-1] = spikes[i+1] .- active_in_traces(i)
            else
                #calculate error from the next layer
                error_by_layer[i-1] = connections[i, i+1] * error_by_layer[i] .* active_in_traces(i)
            end
            delta_by_layer[i-1] = spikes_in_traces(i-1) * error_by_layer[i-1]' .* lr
        end

        #update the weights with the calculated values
        for i in 1:n_layers-1
            connections[i, i+1] .+= delta_by_layer[i]
        end
    end

    net.train_step += 1

    return error_by_layer
end
