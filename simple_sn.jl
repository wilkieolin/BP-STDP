module SNetwork

using Distributions
export Network, update!, update_weights!, run!, train!

    mutable struct Network
        spikes::Dict{Int, Array{Bool,2}}
        potentials::Dict{Int, Array{<:Real,2}}
        connections::Dict{Tuple{Int,Int}, Array{<:Real,2}}

        net_shape::Array{Int,1}
        input_rates::Array{<:Real,1}
        n_layers::Int

        threshold::Real
        reset::Real
        learn_rate::Real
    end

    function Network(net_shape::Array{<:Int,1})
        n_layers = length(net_shape)
        if (n_layers < 2)
            error("Must have at least 2 layers to make a network.")
        end

        #declare the space for the network's variables/objects
        spikes = Dict{Int, Array{Bool,2}}()
        potentials = Dict{Int, Array{Float64,2}}()
        connections = Dict{Tuple{Int,Int}, Array{Float64,2}}()

        rand_dist = Normal(0.1,1)

        #setup the input layer
        n_inputs = net_shape[1]
        input_rates = zeros(Float64, n_inputs)
        potentials[1] = -1 .* ones(Float64, n_inputs, 1)
        spikes[1] = falses(n_inputs, 1)

        for dst in 2:n_layers
            src = dst - 1
            layer_size = net_shape[dst]
            potentials[dst] = zeros(Float64, layer_size, 1)
            spikes[dst] = falses(layer_size, 1)
            connections[(src, dst)] = rand(rand_dist, net_shape[src], net_shape[dst])
        end

        return Network(spikes, potentials, connections,
            net_shape, input_rates, n_layers,
            1.0, 0.0, 0.0005)
    end

    function update!(net::Network)
        spikes = net.spikes
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
            spikes[dst] .= potentials[dst] .> net.threshold
            #reset potential of spiking neurons to zero
            potentials[dst][spikes[dst]] .= net.reset
        end

    end

    function update_weights!(net::Network, desired::Array{<:Real,1})
        if length(net.net_shape) != 3
            error("Algorithm currently only works for networks with 1 hidden layer (Input, Hidden, & Output)")
        end

        spikes = net.spikes
        potentials = net.potentials
        connections = net.connections
        lr = net.learn_rate

        error_output = desired .- spikes[3]
        error_hidden = connections[(2,3)] * error_output .* spikes[2]


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

        history = zeros(time, length(spikes[n_layers]))
        output = falses(time, length(spikes[n_layers]))

        for i in 1:time
            update!(net)
            update_weights!(net, desired)
            history[i,:] = potentials[n_layers]
            output[i,:] = spikes[n_layers]
        end

        return (history, output)
    end


end
