include("H2.jl")
include("H2sparse.jl")
include("statetolabel.jl")
#include("diag.jl")
include("diagsparse.jl")
include("expectZ.jl")
include("Calculation of the entanglement entropy.jl")

using Plots
using LaTeXStrings
using SparseArrays
using Arpack
#using ExpmV # Using Expokit instead of ExpmV as the former is faster
using Expokit
using Statistics


N = 21; J = 1 ; h = (sqrt(5)+1)/4 ; g = (sqrt(5)+5)/8; 
E = 0.01; nee = 80; L = 7# nee = 1 
Ls = rand(1:2^N, L) # A vector of labels of L Fock states in a Product State

#Hd = H2(N, J, h, g) # Short for dense Hamiltonian matrix
Hs = H2sparse(N, J, h, g)
#println("Got Hs")

# Writing initial states to time evolve

function FockState(l, N)
    #=
    This function returns a Fock state with a single spin up at site l of N sites, with the rest down.
    =#
    ϕ = zeros(N)
    ϕ[N+1-l] = 1
    ind = statetolabel(ϕ)
    #display(ϕ)

    ψ = zeros(2^N) 
    ψ[ind] = 1
    #ψ[5] = 1
    #ψ[2^(N-2)] = 1
    #ψ[90] = 1

    return ψ, ϕ # The state representation in a 2^N vector and 
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
    integers which are used as labels in a 2^N vector and the state is normalised by
    1/sqrt(L), where L  is the length of Ls

    N.B. Ls being a random number, so L = 1, returns a random Fock state.
    =#
    
    ψ = zeros(2^N)

    for l in 1:L
        #ϕ = labeltostate(A[l], N)
        #locind = statetolabel(ϕ)
        ψ[Ls[l]] = 1
    end

    return ψ / sqrt(L) , Ls
end

#X = diag(Hd)
#Xs = diagsparse(Hs, E, nee)
#display(Xs[2])

#display(histogram(X, bins = range(minimum(X), stop = maximum(X), length = 40), normalize = false, 
#                  label="N = "*string(N), xlabel = "ε", ylabel = " \nNumber of eigenstates\n "))

#ket = sumofeigenstates(Xs[1], N, nee)
#ket = sumofeigenstates(X[2], X[3], 0, 0.2, N) # Use for dense Hamiltonians
ket = ProductState(N, Ls, L)
#ket = FockState(4, N)

#println("initial state defined")
#bra = conj(ket')
# println(bra*ket) # verifies normalisation

# To calculate the expectation value, can just use expectZ passing U * ket as the vector v
#println(expectZ(ket, 2, N)) # t=0 (initial value)
#expectZ(ψt, 4, N)

function Opt(Ψ, Φ, H, N, i, tmax)
    #=
    This function generates a plot for the evolution of a quantum system of N sites 
    with a given initial state Ψ and Hamiltonian H up to a time tmax for the Z operator
    acting on site i.
    =#
    dτ = tmax/100
    x = 0:dτ:tmax; y = [expectZ(Ψ, i, N)] # 140 gives the number the points; an abitrary choice which could be changed
    
    L = length(x)-1; a = Int64(L/2)
    #println(" \nFor N = ", N)
    ϕ = Ψ
    for τ in 1:L
        ψτ = expmv(-im*dτ, H, ϕ)
        #ψτ = expmv(-im*τ*dτ, H, Ψ)
        locval = expectZ(ψτ, i, N)
        append!(y, locval)
        ϕ = ψτ # Updates the state after each time evolution step.
        #=
        if mod(τ, 10) == 0
            println(τ)
        end=#
    end

    #= # Applicable for Fock States; Use Ls in place of Φ for ket[2]
    S = ""
    for j in string.(Int64.(Φ))
        S =  S * j
    end=#

    graph = plot(x, y, label = "N = "*string(N),
                 title = L"⟨Ψ(t)|Z_%$i|Ψ(t)⟩"*" vs t")# For Fock States: ... vs t for \n"*L"|Ψ(t=0)⟩ = |%$S⟩"
    #println("Making graph")
    xlabel!(L"t")
    ylabel!(" \n "*L"⟨Ψ(t)|Z_%$i|Ψ(t)⟩") # %$ symbol allows for string interpolation so i is correctly presented in the yaxis label
    display(graph)

    σ2 = 1/a * cov(y[a:L], corrected = false)
    println(" \nVariance for N = ", N, " is ", σ2, "\nfor Ls = ", Φ)
end 

function OptEE(Ψ, Φ, H, N, i, tmax)
    #=
    This function generates a plot for the evolution of a quantum system of N sites 
    with a given initial state Ψ and Hamiltonian H up to a time tmax for the Z operator
    acting on site i.
    =#
    dτ = tmax/60# sites = siteinds(2,N)
    x = 0:dτ:tmax; y = [EntEnt2(Ψ, i, N)] 
    
    L = length(x)-1; a = Int64(L/2)
    #println(" \nFor N = ", N)
    ϕ = Ψ
    for τ in 1:L
        ψτ = expmv(-im*dτ, H, ϕ)
        #ψτ = expmv(-im*τ*dτ, H, Ψ)
        locval = EntEnt2(ψτ, i, N)
        #println("locval type is ", typeof(locval))
        append!(y, locval)
        ϕ = ψτ # Updates the state after each time evolution step.
        #=
        if mod(τ, 10) == 0
            println(τ)
        end=#
    end

    S = ""
    for j in string.(Int64.(Φ))
        S =  S * j * ", "
    end
    maxval = i * log(2) - (4^i)/(2^(N + 1))
    y2 = maxval * ones(Int64, length(x))
    maxvalplt = round(maxval, digits=5)

    graph = plot(x, y, label = "N = "*string(N)*" and \nbipartition site "*string(i),
                 title = "Entanglement Entropy vs t for \n"*L"|Ψ(t=0)⟩ = |%$S⟩"*"\n ",legendposition=:bottomright)
    plot!(x, y2, label="Maximal expectation \nvalue for EE = $maxvalplt",ls=:dash, lc="red")
    #println("Making graph")
    xlabel!(L"t")
    ylabel!(" \n "*"Entanglement Entropy\n ") # %$ symbol allows for string interpolation so i is correctly presented in the yaxis label
    display(graph)

    σ2 = 1/a * cov(y[a:L], corrected = false)
    μ = mean(y[a:L])
    println(" \nVariance for N = ", N, " is ", σ2)
    println("And mean equilibrium value is ", μ, " \nand maximum value is ", maxval)
    # return μ, σ2
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
    #println("Making graph\n ")

    graph = plot(x, y, label = "N = "*string(N)*" \nDense H")
    xlabel!(L"t")
    ylabel!(" \n "*L"⟨Ψ(t)|Z_%$i|Ψ(t)⟩") # %$ symbol allows for string interpolation so i is correctly presented in the yaxis label
    display(graph)
end 

function OptEEia(Ψ, Φ, H, N, ia, tmax)
    #=
    This function generates a plot for the evolution of a quantum system of N sites 
    with a given initial state Ψ and Hamiltonian H up to a time tmax for the Entanglement
    Entropy of a subsystem from site 1 to site i out of an array ia with different sites 
    being evaluated as the partition between the 2 subsystems.
    =#
    dτ = tmax/60# sites = siteinds(2,N)
    x = 0:dτ:tmax 
    #y1 = [EntEnt2(Ψ, ia[1], N)]; y2 = [EntEnt2(Ψ, ia[2], N)] y3 = [EntEnt2(Ψ, ia[4], N)]
    #y4 = [EntEnt2(Ψ, ia[4], N)]; y5 = [EntEnt2(Ψ, ia[5], N)]; Y = [y1, y2, y3, y4, y5]; 
    Y = zeros(Float64, (1, length(ia)))
    #display(Y)
    for j in 1:lastindex(ia)
        yl = EntEnt2(Ψ, ia[j], N)
        Y[j] += yl
    end
    #display(Y)
    #println(typeof(Y))

    L = length(x)-1; a = Int64(L/2)
    #println(" \nFor N = ", N)
    ϕ = Ψ
    for τ in 1:L
        ψτ = expmv(-im*dτ, H, ϕ)
        #ψτ = expmv(-im*τ*dτ, H, Ψ)
        for j in 1:lastindex(ia)
            locval = EntEnt2(ψτ, ia[j], N)
            append!(Y[j], locval)
        end
        ϕ = ψτ # Updates the state after each time evolution step.
        #=
        if mod(τ, 10) == 0
            println(τ)
        end=#
    end

    S = ""
    for j in string.(Int64.(Φ))
        S =  S * j * ", "
    end

    for j in 1:lastindex(ia)
        maxval = ia[j] * log(2) - (4^ia[j])/(2^(N + 1))
        ymaxj = maxval * ones(Int64, length(x))
        maxvalplt = round(maxval, digits=5)

        graph = plot(x, Y[j], label = "N = "*string(N)*" and \nbipartition site "*string(ia[j]),
                 title = "Entanglement Entropy vs t for \n"*L"|Ψ(t=0)⟩ = |%$S⟩"*"\n ",legendposition=:bottomright)
        plot!(x, ymaxj, label="Maximal expectation \nvalue for EE = $maxvalplt",ls=:dash, lc="red")
        #println("Making graph")
        xlabel!(L"t")
        ylabel!(" \n "*"Entanglement Entropy\n ") # %$ symbol allows for string interpolation so i is correctly presented in the yaxis label
        display(graph)
        
        σ2 = 1/a * cov(Y[j][a:L], corrected = false)
        μ = mean(Y[j][a:L])
        println(" \nVariance for N = ", N, " is ", σ2)
        println("And mean equilibrium value is ", μ, " \nand maximum value is ", maxval)
    end

    # return μ, σ2
end

amax = round(Int64, N/2 -1)
A = 4:2:amax
OptEEia(ket[1], ket[2], Hs, N, A, 200) # where ket is a Fock state expressed as a vector in the Hilbert space and the spins on local sites.
#println("Completed plot for N = ", N)
#OptD(ket, Hd, N, 4, 600) # For comparison with sparse H using the same state.
#=
for Nloc in 15:2:21
    Hsloc = H2sparse(Nloc, J, h, g)
    #Lsloc = rand(1:2^Nloc, L)
    #ketl = ProductState(Nloc, Lsloc, L) # ketl just sounded better than ketloc
    ketl = FockState(4, Nloc)
    #println("Got Hs and initial state for ", Nloc)

    OptEE(ketl[1], ketl[2], Hsloc, Nloc, round(Int64, Nloc/2), 200)
end=#

# Comparison between entanglement entropy functions
#sites = siteinds(2,N)
#=
X = 3:10; Y1 = []; Y2 = []
for al in X
    println(al)
    #y1 = EntangleEntropy(ket[1], al, sites)
    y1 = EntEnt2(ket[1], al, N)
    y2 = al * log(2) - (4^al)/(2^(N + 1))
    push!(Y1, y1)
    push!(Y2, y2)
end
println(Ls)

comparison_plot = scatter(X, Y1, markershape=:x, label="EntEnt2 function") # Using ITensor
scatter!(X, Y2, markershape=:+, markersize=6, label="Predicted maximal value") #Using Numpy
xlabel!("Size of the subsystem\n ")
ylabel!(" \nEntanglement entropy measurement\n ")
title!("Comparison between EntEnt2 and predictions in\n Entanglement Entropy calculations for N = $N\n ")
display(comparison_plot)=#
