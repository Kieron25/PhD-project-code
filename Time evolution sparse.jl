include("H2.jl")
include("H2sparse.jl")
include("statetolabel.jl")
#include("diag.jl")
include("diagsparse.jl")
include("expectZ.jl")

using Plots
using LaTeXStrings
using SparseArrays
using Arpack
using ExpmV


N = 11; J = 1 ; h = (sqrt(5)+1)/4 ; g = (sqrt(5)+5)/8; 
E = 0.01; nee = 80; L = 20 # 2^(Int64(round(N/2)+1)) # nee = 1 
Ls = rand(1:2^N, L) # A vector of labels of L Fock states in a Product State
#t  = 40

#Hd = H2(N, J, h, g) # Short for dense Hamiltonian matrix
Hs = H2sparse(N, J, h, g)
#U = exp(-im*Hm*t)
# May have to use incremental times to create a U(t) operator which is made up of tiny U's 
# acting consecutively on  
println("Got Hs")
# Writing initial states to time evolve

function FockState(l, N)
    #=
    This function returns a Fock state with a single spin up at site l of N sites, with the rest down.
    =#
    ϕ = zeros(N)
    ϕ[l] = 1
    ind = statetolabel(ϕ)

    ψ = zeros(2^N) 
    ψ[ind] = 1
    #ψ[5] = 1
    #ψ[2^(N-2)] = 1
    #ψ[90] = 1

    return ψ
end

function lowesteigenstate(H, E)
    #=
    This function returns the lowest energy eigenstate for a given Hamiltonian H using the diag function, 
    which the first output of diagsparse.

    The argument 0 is such that nee == 0 selects the lowest eigenstate, so nee = 1 can be 
    used to pick an energy closest a given energy E without any contradictions.

    This returns the lowest energy eigenvector for a Hamiltonian H. 
    =#
    return diagsparse(H, E, 0)[1]
end

function sumofeigenstates(Mev, N, nee)
    #=
    This function returns a state which is a superposition of eigenstates within a given energy window 
    characterised by E0 - δE < E < E0 + δE.

    Mev is a matrix of eigenvectors given by the 2nd output of diag() and Ee is a vector of Eigenenergies 
    given by the 3rd output of diag(). N has its usual meaning as system size.
    Mev is first output of diagsparse() and Ee is the second.

    I chose to eliminate the filtering code which seemed to add too much time and 
    states are instead counted by being the nee closest in energy to E, given in diagsparse()
    =#
    #=
    A = filter(x -> E0 - δE <= x <= E0 + δE, Ee)
    B = filter(x -> x >= E0 - δE, A)
    a = first(A); b = last(A)
    i1 = indexin(a, Ee)[1]; i2 = indexin(b, Ee)[1]
    println(i1, i2) =#

    denom = sqrt(nee)
    L = 2^N

    Ψ = zeros(L) 
    for α in 1:L # α goes along the rows 
        for β in 1:nee # β goes along the columns
            #println(β)
            # Ψ += Mev[: , β]
            Ψ[α] += Mev[α , β] # Element-wise addition rather than adding vectors together
        end
    end
    return Ψ / denom 
end

function ProductState(N, Ls, L)
    #=
    This function creates a normalised Product State of Fock States using an array A of
    integers which are used as labels and the state is normalised by 1/sqrt(L), where L 
    is the length of Ls
    =#
    
    ψ = zeros(2^N)

    for l in 1:L
        #ϕ = labeltostate(A[l], N)
        #locind = statetolabel(ϕ)
        ψ[Ls[l]] = 1
    end
    return ψ / sqrt(L)
end

#X = diag(Hd)
#Xs = diagsparse(Hs, E, nee)
#display(Xs[2])
#println("Completed diagsparse")
#display(histogram(X, bins = range(minimum(X), stop = maximum(X), length = 40), normalize = false, 
#                  label="N = "*string(N), xlabel = "ε", ylabel = " \nNumber of eigenstates\n "))

#ket = sumofeigenstates(Xs[1], N, nee)
#ket = sumofeigenstates(X[2], X[3], 0, 0.2, N) # Use for dense Hamiltonians
ket = ProductState(N, Ls, L)
println("Initial state defined")
#display(ket)
#bra = conj(ket')
# println(bra*ket) # verifies normalisation

# To calculate the expectation value, can just use expectZ passing U * ket as the vector v
#println(expectZ(ket, 2, N)) # t=0 (initial value)
#expectZ(ψt, 4, N)


function Opt(Ψ, H, N, i, tmax)
    #=
    This function generates a plot for the evolution of a quantum system of N sites 
    with a given initial state Ψ and Hamiltonian H up to a time tmax for the Z operator
    acting on site i.
    =#
    dτ = tmax/60
    x = 0:dτ:tmax; y = [expectZ(Ψ, i, N)] # 140 gives the number the points; an abitrary choice which could be changed
    
    L = length(x)-1
    for τ in 1:L
        ψτ = expmv(-im*τ*dτ, H, Ψ)
        locval = expectZ(ψτ, i, N)
        append!(y, locval)
        println(τ)
    end

    graph = plot(x, y, label = "N = "*string(N)*" \nSparse H")
    println("Making graph")
    xlabel!(L"t")
    ylabel!(" \n "*L"⟨Ψ(t)|Z_%$i|Ψ(t)⟩") # %$ symbol allows for string interpolation so i is correctly presented in the yaxis label
    display(graph)
end 

function OptD(Ψ, H, N, i, tmax)
    #=
    This function generates a plot for the evolution of a quantum system of N sites 
    with a given initial state Ψ and Hamiltonian H up to a time tmax for the Z operator
    acting on site i; 

    This is the same function used to plot the expectation value over time in the 
    dense Hamiltonian time evolution file.
    =#
    dτ = tmax/120
    x = 0:dτ:tmax; y = [expectZ(Ψ, i, N)] # 140 gives the number the points; an abitrary choice which could be changed
    U = exp(-im*dτ*H)

    L = length(x)-1
    for τ in 1:L
        ψτ = U^τ * Ψ 
        locval = expectZ(ψτ, i, N)
        append!(y, locval)
    end
    println("Making graph")

    graph = plot(x, y, label = "N = "*string(N)*" \nDense H")
    xlabel!(L"t")
    ylabel!(" \n "*L"⟨Ψ(t)|Z_%$i|Ψ(t)⟩") # %$ symbol allows for string interpolation so i is correctly presented in the yaxis label
    display(graph)
end 

Opt(ket, Hs, N, 5, 200)
#println("Completed Sparse plot")
#OptD(ket, Hd, N, 4, 600) # For comparison with sparse H using the same state.
println("Done ")