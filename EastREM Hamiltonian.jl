#=
EastREM Hamiltonian construction, to obtain the adjancency matrix for FSA

H = H_0 + V, H_0 is a random on-site energy in the Fock Space lattice 
and V is the interaction that allows hopping between states

=#

# Imported functions from earlier in the project that can be used to construct the EastREM Hamiltonian
include("labeltostate.jl")
include("statetolabel.jl")
include("Xop2.jl")
include("Zop2.jl") 
include("Domain Wall state.jl") # returns the state label for the domain wall state
include("Hamming distance.jl") # hammingdis_2 comes from this
include("transfer_matrix_functions.jl") # a file with some old functions in it

using Random
using Distributions
using SparseArrays
using Plots

# Defining parameters in the model
N = 12; D = 2^N; Γ = 0.1

function statevec(label, D)
    # This function takes a state label to make a sparse vector of length D and appends a 1 to this site.
    # This corresponds to the Fock state at 

    V = sparsevec(Dict(label => 1), D)
    return V
end

function H_East(N, Γ)
    #=
    This function creates the adjacency matrix for the EastREM hamiltonian with periodic boundary conditions.
    =#
    D = 2^N
    
    # Constructing the Hamiltonian with periodic boundary conditions
    #H = zeros(typeof(Γ), D, D)
    Es = zeros(typeof(Γ), D)
    I = Int64[]; J = Int64[]; V = Float64[]

    # And normal distribution the on-site energies are drawn from form a list of values which are indexed
    # from when the diagonal part of the Hamiltonian is included
    Random.seed!(rand(1:10000)) # Having the seed be a random number gives different energies for each seed.
    d = Normal(0, 3)
    dis = rand(d, D)

    for α in 1:D
        Ψ = labeltostate(α, N)
        for j in 1:N
            sf = (1.0 + Zop2(Ψ, mod1(j+1, N)))/2
            ϕ = Xop2(Ψ, j)
            β = statetolabel(ϕ)
            if sf == 1.0
                if α < β
                    #H[α, β] += sf * Γ
                    #H[β, α] += conj(sf * Γ)
                    #denom = Γ/(dis[α]-dis[β])
                    append!(I, α)
                    append!(J, β)
                    append!(V, Γ)
                    # Below interactions are allowed in the Hamiltonian
                    append!(I, β)
                    append!(J, α)
                    append!(V, conj(Γ))
                else
                    #Nothing happens (ensuring hermiticity of the matrix)
                end
            else
                #Nothing happens (only nonzero elements are in the sparse matrix)
            end
        # This line considers on-site energies; for the adjacency matrix, this is commented out since only hopping between 
        # sites is considered in the interacting part of the Hamiltonian
        
        Es = dis[α]
        append!(I, α)
        append!(J, α)
        append!(V, Es)
        end
    
    end
    Hs = sparse(I, J, V)
    return Hs, dis
end

function fullTM(N, Γ)
    #=
    This function creates the complete transfer matrix for the EastREM hamiltonian with periodic boundary conditions.
    
    This is independent of the initial state, so it can be considered for all initial states at random to obtain 
    the probability distribution across all separations and 
    =#
    D = 2^N
    Isl = domainwalllabel(N) # Initial state label
    I = Int64[]; J = Int64[]; V = Float64[]

    # And normal distribution the on-site energies are drawn from form a list of values which are indexed
    # from when the diagonal part of the Hamiltonian is included
    Random.seed!(rand(1:Int64(1E11))) # Having the seed be a random number gives different energies for each seed.
    d = Normal(0, N)
    dis = rand(d, D)
    #A = collect(1:Isl-1)
    #append!(A, collect(Isl+1:D))

    for α in 1:D #A
        Ψ = labeltostate(α, N)
        for j in 1:N
            sf = (1.0 + Zop2(Ψ, mod1(j+1, N)))/2 # returns 1 or 0
            ϕ = Xop2(Ψ, j)
            β = statetolabel(ϕ)
            if sf == 1.0
                if β == Isl
                    append!(I, β)
                    append!(J, α)
                    append!(V, 1.0) # Unsure if this should be set to zero or one.
                    # This is the initial state so it shouldn't return to this state
                    
                else
                    denom = Γ/(dis[Isl]-dis[β])
                    append!(I, β)
                    append!(J, α)
                    append!(V, denom)
                    #Nothing happens (ensuring hermiticity of the matrix)
                end
            else
                #Nothing happens (only nonzero elements are in the sparse matrix)
            end
        end
    # This line considers on-site energies; for the adjacency matrix, this is commented out since only hopping between 
    # sites is considered in the interacting part of the Hamiltonian
    #Es[α] += dis[α]
    
    end
    Hs = sparse(I, J, V)
    return Hs, dis
end

#display(H_East(4)[1])
#println(zeros(D,1))

function transfer_amp3(N, Γ, r, init)
    #=
    A function to compute the transfer amplitude using the hammingdis_2 function

    This function gets around the issues of previous functions by hammingdis_2 using a while loop#
    so the shortest path and only the shortest path count towards the amplitude of a site to state
    and previously visited states aren't considered by the condition of the while loop.
    =#
    D = 2^N
    M = fullTM(N, Γ)[1]
    rs = []; os = [] # an array of hamming distances and overlaps, in order of the state label
    for k in 2:D
        rl, ol = hammingdis_2(init, k, D, M)
        #println(k, " , ", rl, " , ", ol)
        append!(rs, rl); append!(os, ol)
    end

    indarr = findall(x -> x == r, rs) # tells you the states at the hamming distance r
    r_overlap = []
    for l in indarr
        push!(r_overlap, os[l])
    end
    return maximum(r_overlap)
end

function maxamp2(N, Γ, init)
    #=
    This functions finds the maximum value in a vector and returns the amplitude.
    =#
    #V = transfer_amp2(N, Γ, N, init)
    #maxval = maximum(V)
    maxval = transfer_amp3(N, Γ, N, init)
    #return log(maximum(abs.(maxval))^2)/(2*N) - log(Γ)
    return (log((maxval)^2)/(2*N)) - log(Γ) + 1
end

function histogramforN(N, Γ)
    #=
    This function plots a histogram of maximum amplitudes for a given N and Γ, to reproduce figure 11a in 
    Achilleas and Sthithadhi's paper. 

    It returns the bin values so it can make up the multi-figure plot.
    If this function isn't being run, the plot can be returned.
    =#

    binvals = []; initlab = domainwalllabel(N)
    for j in 1:5000
        #append!(binvals, maxamp(N, Γ, N)) # using maxamp function
        append!(binvals, maxamp2(N, Γ, initlab))
    end
    #binrange = range(minimum(binvals), stop = maximum(binvals), length=25)
    #=
    # These lines take the above data and return a single distribution plot.
    plot1 = stephist(binvals, normalize=:pdf,  label="N = $N", yscale=:log10, legend=:topleft)
    xlabel!("Λ_$N"); ylabel!("P(Λ_$N)")
    xlims!(-2,3)
    ylims!(1E-3,1)
    =#
    
    #display(plot1)
    return binvals # plot1
end

function multiplothist(Nmin, Nmax, Γ)
    #=
    This function plots multiple distributions on the same axes for ease of comparison.

    This is pre-written for 5 values of N, of a separation 2 between the smallest and largest system sizes.
    =#
    
    bN = Nmin+2; cN = Nmin+4; dN = Nmax - 2
    a, b, c, d, e = histogramforN(Nmin, Γ), histogramforN(bN, Γ), histogramforN(cN, Γ), histogramforN(dN, Γ), histogramforN(Nmax, Γ) #c, d, e =  histogramforN(cN, Γ), histogramforN(dN, Γ), histogramforN(Nmax, Γ) 
    stephist([a b c d e], label = ["N = $Nmin" "N = $bN" "N = $cN" "N = $dN" "N = $Nmax"], normalize=:pdf, yscale=:log10, legend=:topright, lw = [3 3 3 3 3]) # c d e , # in label: "N  = $cN" "N  = $dN""N = $Nmax"
    xlabel!("Λ_N \n "); ylabel!(" \n P(Λ_N)") 
    xlims!(-2,3)
    ylims!(1.4E-3,1.4E0)
    #yticks!(1E-3:?:1, ["1E-3", "1E-2", "1E-1", "1E0"])
end


#M = fullTM(3, Γ)[1]; 
#display(M)
#init = domainwalllabel(N)
#transfer_amp3(N, Γ, N, init)
#transfer_amp2(N, Γ, N, init)
#histogramforN(N, Γ)
#SV = sparsevector_i(M, init, N, D)
#multiplothist(6, 14, Γ)
#display(heatmap(Array(H_East(5, 30)[1])))

#=
for Nl in 10:2:16
    histogramforN(Nl, Γ)
end=#

#=
# Old code to test the functions
#display(H_East(N, Γ)[1])
#first = 5; r = 6
first = domainwalllabel(N); r = 8
Trm = TMfr(N, H_East(N, Γ)[1], H_East(N, Γ)[2], first, r, Γ^(r-1)) # initial state label is first, going to states r away
#println(Trm)
println(maximum(Trm))

# TMfr could then be used to compile a function of amplitudes 

#M = fullTM(N, Γ)[1]; r = 3
#val = transfer_amp(N, M, r, 3, 7)

=#