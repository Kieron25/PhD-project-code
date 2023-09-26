#=
This file contains the functions to compute the Block IPR for a given initial state
localised to a block; for comparison with previous work, the state with half spins up and 
half spins down is selected corresponding to a block in the middle of the matrix, 
referred to in the literature as the Domain Wall State.
=#

using Arpack
using LinearAlgebra

# This file has the functions for the GOE EastREM matrix and other useful files and modules loaded.
include("GOE chain matrix.jl") 
include("Domain Wall state.jl")
include("EastREM Hamiltonian.jl")

function TotalD(N)
    #=
    L = [1]
    for j in 1:N-1 
        append!(L, 2^j)
    end
    return sum(L)
    =#
    return 2^N - 1
end

function eigenvalsandvecs(M, D)
    # This function returns D - 1 eigenvalues λ and eigenvectors Ψ for a matrix M
    #
    # The eigenvectors are returned in a matrix, where Ψ[:, i] is the i-th eigenvector
    # with an eigenvalue λ[i]

    λ, Ψ = eigs(M, nev = D - 1)
    return λ, Ψ
end

function overlapvalue(i, V)
    #=
    For a normalised vector with a single value of 1 at site i,
    the overlap between this vector and another vector V will be the i-th element 
    of V, so the i-th eleemnt is selected to get around multiplying vectors.

    The modulus square of the overlap value is returned since this is the important quantity
    in the sum. 
    Vector V is an eigenvector, and i is the domainwall label to find the overlap between 
    the initial state and the eigenstates, or it may be the label of states within blocks
    =#
    val = V[i]
    return conj(val) * val
end

function BlockIPR(N, M)
    Ik = 0; K = Int64(N/2)
    αmin = TotalD(K); αmax = TotalD(K+1); D = TotalD(N)
    L = domainwalllabel(N)

    λ, Ψ = eigenvalsandvecs(M, D)

    for α in αmin:αmax
        for n in 1:D-1
            c_n = overlapvalue(L, Ψ[:, n])
            term2 = Ψ[:, n][α] # pulls out the α element of the eigenvector.
            locval = c_n * conj(term2) * term2
            Ik += locval
        end
    end
    #println("Ik = $Ik")
    return Ik
end
function AverageIk(N, Γ, Nit)
    Ik = 0
    for j in 1:Nit
        Ml = GOEEastREMM(N, Γ)[1]
        Ik += BlockIPR(N, Ml)
    end
    return Ik/Nit
end

function AverageIk2(N, Γ, Nit)
    Ik = 0
    for j in 1:Nit
        Ml = H_East(N, Γ)[1]
        Ik += BlockIPR(N, Ml)
    end
    return Ik/Nit
end
#N = 6; Γ = 0.1
#M = GOEEastREMM(N, Γ)[1]
#BlockIPR(N, M)

#AverageIk(N, Γ, 70)

function IkvsΓ(Nmin, Nmax, Γmin, Γmax)
    #=
    This function combines the previous work to create a plot of the Block IPR across 
    different hopping strengths Γ, across a small range of system sizes N - here N is 
    the number of blocks which is the same as the number of spins in the system.

    Here, Γmin & Γmax are defined as the minimum and maximum indices such that the hopping
    parameter varies from 10^Γmin to 10^Γmax 
    =#
    Ik1 = []; X = 10 .^ collect(range(Γmin, Γmax, 11))
    for Γ in X
        append!(Ik1, AverageIk(Nmin, Γ, 60))
    end
    graph = plot(X, Ik1, label="N = $Nmin", xscale=:log10, markershape=:xcross, legendposition=:topright)
    title!("Block IPR for domain wall state vs Γ \n for different numbers of blocks, N")
    xlabel!("Γ\n "); ylabel!(" \nI_k")
    println("N = $Nmin completed")

    for N in Nmin+2:2:Nmax
        IkA = []
        for Γ in X
            append!(IkA, AverageIk(N, Γ, 40))
        end
        plot!(X, IkA, label="N = $N", markershape=:xcross, legendposition=:topright)
        println("N = $N completed")
    end

    display(graph)
end

#IkvsΓ(6, 10, -2, 1)
#display(heatmap(Matrix(GOEEastREMM(5, 20)[1])))
#display(heatmap(Matrix(H_East(5, 40)[1])))