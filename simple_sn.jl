module SNetwork

using Distributions
export Network, update!, update_weights!, run!, train!

mutable struct Network
    spikes::Dict{Int, Array{Bool,2}}
    traces::Dict{Int, Array{Bool,2}}
    potentials::Dict{Int, Array{<:Real,2}}
    connections::Dict{Tuple{Int,Int}, Array{<:Real,2}}

    net_shape::Array{Int,1}
    input_rates::Array{<:Real,1}
    n_layers::Int

    thresholds::Array{<:Real,1}
    reset::Real
    memory::Int
    step::Int
    learn_rate::Real
end

function Network(net_shape::Array{<:Int,1}, mean::Real, std::Real, memory::Int)
    n_layers = length(net_shape)
    if (n_layers < 2)
        error("Must have at least 2 layers to make a network.")
    end

    #declare the space for the network's variables/objects
    spikes = Dict{Int, Array{Bool,2}}()
    traces = Dict{Int, Array{Bool,3}}()
    potentials = Dict{Int, Array{Float64,2}}()
    connections = Dict{Tuple{Int,Int}, Array{Float64,2}}()

    rand_dist = Normal(mean, std)

    #setup the input layer
    n_inputs = net_shape[1]
    input_rates = zeros(Float64, n_inputs)
    potentials[1] = -1 .* ones(Float64, n_inputs, 1)
    spikes[1] = falses(n_inputs, 1)
    traces[1] = falses(n_inputs, memory)

    for dst in 2:n_layers
        src = dst - 1
        layer_size = net_shape[dst]
        potentials[dst] = zeros(Float64, layer_size, 1)
        spikes[dst] = falses(layer_size, 1)
        traces[dst] = falses(layer_size, memory)
        connections[(src, dst)] = rand(rand_dist, net_shape[src], net_shape[dst])
    end

    return Network(spikes, potentials, connections,
        net_shape, input_rates, n_layers,
        ones(n_layers), 0.0, memory, 0, 0.0005)
end

function update!(net::Network)
    spikes = net.spikes
    memory = net.memory
    potentials = net.potentials
    connections = net.connections

    #calculate random spiking at the inputs
    spikes[1][:,1] = rand(net.net_shape[1]) .< net.input_rates

    for i in 2:net.n_layers
        src = i - 1
        dst = i

        #add the charge from spikes in the last layer
        currents = transpose(transpose(spikes[src]) * connections[(src, dst)])
        potentials[dst] .+= currents
        #did this cause spikes?
        spikes[dst] .= potentials[dst] .> net.thresholds[dst]
        net.step += 1
        #update the synaptic trace
        ind = mod1(net.step, net.memory)
        memory[:,ind] = spikes[dst]
        #reset potential of spiking neurons to zero
        potentials[dst][spikes[dst]] .= net.reset
    end

end

function update_weights!(net::Network, desired::Array{<:Real,1})
    if length(net.net_shape) != 3
        error("Algorithm currently only works for networks with 1 hidden layer (Input, Hidden, & Output)")
    end

    spikes = net.spikes
    memory = net.memory
    potentials = net.potentials
    connections = net.connections
    lr = net.learn_rate

    spike_in_memory(x::Int) = (sum(memory[3], dims=2) .> 1)

    error_output = desired .- spike_in_memory(3)
    error_hidden = connections[(2,3)] * error_output .* spike_in_memory(2)


    deltas_23 = spikes[2] * error_output' .* lr
    deltas_12 = spikes[1] * error_hidden' .* lr

    connections[(2,3)] .+= deltas_23
    connections[(1,2)] .+= deltas_12

    return (deltas_12, deltas_23)
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

function train!(net::Network, time::Int, desired::Array{<:Real,1})
    spikes = net.spikes
    potentials = net.potentials
    connections = net.connections
    n_layers = net.n_layers

    history = Dict{Int, Array{<:Real,2}}()
    output = Dict{Int, Array{Bool,2}}()
    weights = Dict{Int, Array{<:Real, 3}}()

    for i in 1:n_layers
        history[i] = zeros(time, net.net_shape[i])
        output[i] = falses(time, net.net_shape[i])
        if i != n_layers
            weights[i] = zeros(time, net.net_shape[i], net.net_shape[i+1])
        end
    end

    for i in 1:time
        update!(net)
        update_weights!(net, desired)
        for j in 1:n_layers
            history[j][i,:] = potentials[j]
            output[j][i,:] = spikes[j]
            if j != n_layers
                weights[j][i,:,:] = connections[(j, j+1)]
            end
        end
    end

    return (history, output, weights)
end

function epoch(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int)
    (n_x, s_x) = size(x)
    (n_y, s_y) = size(y)

    if s_x != net.net_shape[1]
        error("Second dimension of examples must match input dimension of network")
    elseif n_x != n_y
        error("Each training example (x) must have a target firing rate (y)")
    elseif s_x != net.net_shape[net.n_layers]
        error("Second dimension of target firing rate must match output dimension of network")
    end

    time_per_example = floor(Int, time/n_x)
    for i in 1:n_x
        #set the input firing rate to the example
        net.input_rates = x[i,:]
        #train the network with the desired firing rate
        (h,o,w) = train!(net, time_per_example, y[i,:])
    end

end

function test(net::Network, x::Array{<:Real,2}, y::Array{<:Real,2}, time::Int)
    (n_x, s_x) = size(x)
    (n_y, s_y) = size(y)

    if s_x != net.net_shape[1]
        error("Second dimension of examples must match input dimension of network")
    elseif n_x != n_y
        error("Each training example (x) must have a target firing rate (y)")
    elseif s_x != net.net_shape[net.n_layers]
        error("Second dimension of target firing rate must match output dimension of network")
    end

    time_per_example = floor(Int, time/n_x)
    rates = zeros(n_x, net.net_shape[end])

    for i in 1:n_x
        #see how the network responds to the example stimulus
        net.input_rates = x[i,:]
        (v,s) = run!(net, time_per_example)
        #get the output firing rate
        rates[i,:] = vec(sum(s[net.n_layers], dims=1) ./ time_per_example)
    end

    return rates
end

end
