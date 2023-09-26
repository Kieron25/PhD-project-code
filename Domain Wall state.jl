# Domain Wall state
#=
In the EastREM Hamiltonian, starting from a Domain wall state is optimal since this state is in the bulk of the space - 
it will take the most steps to arrive at a state where the island of down spins isn't frozen.

To this end, I take a system of size N and define the domain wall state label from the spin configuration.

Starting from the right to N/2, spins are up and the remaining ones are down.
For even N, half are up and rest are down; for odd N, 1 more spin is down than up.
=#

function domainwalllabel(N)
    bound = Int64(round((N-0.2)/2))
    #println(bound)
    label = 1 # offset by 1 due to all spins down having a label = 1, 
    for i in 1:bound
        label += 2^(i-1) # The rightmost spin-up in a chain adds a value 2^i
    end
    return label
end

function statevec(label, D)
    # This function takes a state label to make a sparse vector of length D and appends a 1 to this site.
    # This corresponds to the Fock state at 

    V = sparsevec(Dict(label => 1), D)
    return V
end

# domainwalllabel(5)
