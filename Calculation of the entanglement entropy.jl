# A calculation of the entanglement entropy
#=
From the Many-Body vector describing the state Ψ, of dimension 2^N where N is the number 
of sites in the system, the code returns the entanglement entropy of this state.
=#

using ITensors
ITensors.disable_warn_order()
using PyCall
np = pyimport("numpy")

#=
include("statetolabel.jl")

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

function ProductState(N, Ls, L)
    #=
    This function creates a normalised Product State of Fock States using an array A of
    integers which are used as labels in a 2^N vector and the state is normalised by
    1/sqrt(L), where L  is the length of Ls
    =#
    
    ψ = zeros(2^N)

    for l in 1:L
        #ϕ = labeltostate(A[l], N)
        #locind = statetolabel(ϕ)
        ψ[Ls[l]] = 1
    end
    return ψ / sqrt(L)
end
=#
#=
function EntangleEntropy(Ψ, b, sites)
    #=
    The function takes Ψ and changes it to an MPS, and then calculates and
    returns the entanglement entropy of the system between regions A and B, 
    where site b is in the middle of these regions - this is parsed as the 
    variable i following conventions from the earlier code. Ψ is a vector of 
    length 2^N. 

    This function performs the same purpose as ExpectZ to measure a quantity for 
    a given wave function.

    NOTE: This function has been replaced with EntEnt below but both produce the same results.
    =#
    # sites = siteinds(2,N)
    cutoff = 1E-7; maxdim = 10
    M = MPS(Ψ, sites; cutoff=cutoff, maxdim=maxdim)

    orthogonalize!(M, b)
    U,S,V = svd(M[b], (linkind(M, b-1), siteind(M,b)))
    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end

    return SvN 
end=#

function EntEnt(psi, b, N)
    #=
    Sample code from Achilleas to try and measure the entanglement entropy for a given state vector psi.

    This splits it into 2 equal parts; in principle, this could be changed with some extra arguments.
    =#
    #block_dim = Int64(np.sqrt(size(psi)[1])) # Had psi.shape[0] but Julia indexes from 1.
    block_dim1 = 2^b; block_dim2 = 2^(N-b)
    #psi_block = np.reshape(psi, (block_dim, block_dim))
    psi_block = np.reshape(psi, (block_dim2, block_dim1))
    s = np.linalg.svd(psi_block, compute_uv=0)
    sa = s[s .> 1e-15].^2
    return -np.inner(np.log(sa), sa)
end
#=
N = 16; b = 7; L = 10
Ls = rand(1:2^N, L)
#ket = FockState(3, N)[1]
ket = ProductState(N, Ls, L)
EntangleEntropy(ket, b, N)
=#
#=
xa = [9:22]; ya = log2.([7.9433e-6, 8.65375e-6, 1.07772e-5, 7.439e-6, 4.737e-6, 1.70986e-6, 6.1183e-7, 2.5114e-7, 
                         8.25011e-8, 3.6058e-8, 1.286e-8, 3.3585e-9, 1.136e-9, 3.5437e-10])
yμ = log2.([1.95, 1.698, 1.367, 1.008, 0.6908, 0.4542, 0.2719, 0.1662, 0.0962, 0.055, 0.0311, 0.0174, 0.0095676, 0.00524])
graph = scatter(xa, ya, title = " \n"*L"log_2("*" Variance of EE "*L")"*" for N ϵ [9, 22]", legend=false)
xlabel!(L"N")
ylabel!(" \n"*L"log_2("*" Variance of EE "*L")") # %$ symbol allows for string interpolation so i is correctly presented in the yaxis label
display(graph)

graphμ = scatter(xa, yμ, title = " \n"*L"log_2("*" long-time mean of EE "*L")"*" for N ϵ [9, 22]\n ", legend=false)
xlabel!(L"N")
ylabel!(" \n"*L"log_2("*" long-time mean of EE "*L")") # %$ symbol allows for string interpolation so i is correctly presented in the yaxis label
display(graphμ)=#