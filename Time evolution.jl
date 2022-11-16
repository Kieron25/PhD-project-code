include("H2.jl")
include("H2sparse.jl")
include("statetolabel.jl")
include("diag.jl")
#include("diagsparse.jl")
include("expectZ.jl")

using Plots
using LaTeXStrings
#using SparseArrays
#using ExpmV


N = 11; J = 1 ; h = (sqrt(5)+1)/4 ; g = (sqrt(5)+5)/8 
E = 0.01; nee = 80
#t  = 40

Hm = H2(N, J, h, g) # Short for Hamiltonian matrix
Hs = H2sparse(N, J, h, g)
#U = exp(-im*Hm*t)
# May have to use incremental times to create a U(t) operator which is made up of tiny U's 
# acting consecutively on  

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

    return ψ
end

function lowesteigenstate(H)
    #=
    This function returns the lowest energy eigenstate for a given Hamiltonian H using the diag function, 
    which the first column of the second output is the lowest energy eigenstate (the ground state for H).
    =#
    return diag(H)[2][:,1]
end

function sumofeigenstates(Mev, Ee, E0, δE, N)
    #=
    This function returns a state which is a superposition of eigenstates within a given energy window 
    characterised by E0 - δE < E < E0 + δE.

    Mev is a matrix of eigenvectors given by the 2nd output of diag() and Ee is a vector of Eigenenergies 
    given by the 3rd output of diag(). N has its usual meaning as system size.
    =#
    A = filter(x -> E0 - δE <= x <= E0 + δE, Ee)
    #B = filter(x -> x >= E0 - δE, A)
    a = first(A); b = last(A)

    i1 = indexin(a, Ee)[1]; i2 = indexin(b, Ee)[1]

    denom = sqrt(length(A))

    Ψ = zeros(2^N)
    for j in i1:1:i2
        Ψ += Mev[:, j]
    end
    return Ψ / denom
end


##X = diag(Hm)
Xs = diagsparse(Hs, E, nee)
#display(X[3])
#display(histogram(X, bins = range(minimum(X), stop = maximum(X), length = 40), normalize = false, 
#                  label="N = "*string(N), xlabel = "ε", ylabel = " \nNumber of eigenstates\n "))

#ket = sumofeigenstates(X[2], X[3], 0, 0.2, N)
ket = sumofeigenstates(Xs[1], Xs[2], 0, 0.2, N)
#ket = FockState(5, N)
#display(ket)
#bra = conj(ket')
# println(bra*ket) # verifies normalisation
#display(U * ψ)
#ψt = U * ket
# To calculate the expectation value, can just use expectZ passing U * ket as the vector v
#println(expectZ(ket, 2, N)) # t=0 (initial value)
#expectZ(ψt, 4, N)


function Opt(Ψ, H, N, i, tmax)
    #=
    This function generates a plot for the evolution of a quantum system of N sites 
    with a given initial state Ψ and Hamiltonian H up to a time tmax for the Z operator
    acting on site i; 
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

Opt(ket, Hm, N, 4, 600)
println("Done!")