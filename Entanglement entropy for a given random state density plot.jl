# Generating a plot of random state density vs entanglement entropy 
# of the random state of a given density p.

include("statetolabel.jl")
include("Calculation of the entanglement entropy.jl")
include("diagsparse.jl")
include("multi-graph plot.jl")

using Plots
using Statistics
using SparseArrays
using Random
using StatsBase

# System size is fixed
N = 15; b = 7; E = 0.1; p = 0.05
#D = 2^N

function RandArr(N, p)
    #A = Int64[]
    L = Int64(round((p) * 2^N))
    #=
    if L == 0
        # i.e. when p < 2^(-N-1) so round(...) < 0.5 and it's more probable for 
        # all entries to be zero on average
        A = zeros(2^N)
        println("Most probable that no states are involved: p = $p < 2^(N +1 = $N + 1)")
    else
        for i in 1:L
            a = Int64(round(1 + (i - 1) * 2^N / L))
            b = Int64(round(i * 2^N / L))
            Ls = rand(a:b, 1) 
            append!(A, Ls)
        end
    end
    =#
    
    A = sample(1:2^N, L, replace = false) # Samples without replacement
    return A
end

function ProductState(D, Ls)
    #=
    This function creates a normalised Product State of Fock States using an array A of
    integers which are used as labels in a 2^N vector and the state is normalised by
    1/sqrt(L), where L  is the length of Ls.

    N.B. Ls being a random array of numbers, so L = 1, returns a random Fock state.
    =#
    
    ψ = zeros(ComplexF64, D)

    for l in Ls
        #ϕ = labeltostate(A[l], N)
        #locind = statetolabel(ϕ)
        ψ[l] = 1 * exp(im * rand(pi/15:pi/15:2pi))
    end

    norm = sqrt(count(!iszero, ψ))
    return ψ / norm #, Ls
end

function RandomState(p, D) 
    pa = AbstractFloat(p)
    Ψ = sprand(D, 1, pa)
    Ψr = round.(Ψ) # Note: may need to change element type to integers at some point
    norm = sqrt(count(!iszero, Ψr))
    dv = Array(Ψr)/norm
    return dv
end

function pvsEEpoint(p, D, N, b)
    EE = []; len = 100
    #L = Int64(round(p * D))
    #rng = MersenneTwister(1234)

    for j in 1:len
        Ls = RandArr(N, p)
        #Ls = randsubseq(rng, 1:D, p)
        Ψl = ProductState(D, Ls)
        #Ψl = RandomState(p, D)
        EEl = EntEnt2(Ψl, b, N)
        append!(EE,EEl)
    end

    val = sum(EE)/(len)
    #var = cov(EE, corrected = false)
    #expected = b * log(2) - 4^b/(2^(N+1))
    #δ = expected - val
    return val#, var
end
#=
for P in 0.1:0.2:0.9
    println(P)
    println(pvsEEpoint(P, D, N, b), "\n ")
end=#
#println(EntEnt2(ProductState(D, RandArr(N, p)), b, N))
#println(p, " , ",pvsEEpoint(p, D, N, b))

function pvsEEplot(N, b)
    D = 2^N
    #Pa = collect(range(-log(D), log(8)-log(D), 40))
    #Pa = exp.(range(1/D, 8/D, 100))
    #Pb = collect(range(-6, 0, 40))
    #Pb = exp.(range(12/D, 1, 40))
    #P = append!(Pa, Pb)
    P = collect(range(-log(D), 0, 120))
    #Pexp = exp.(P)

    ES = []; #ESeb = [] # Error for Entanglement entropy (S) and its error bars

    for p in P
        val = pvsEEpoint(exp(p), D, N, b)
        #esl = val#[1]
        #esebl = sqrt(val[2])
        append!(ES, val)
        #append!(ESeb, esebl)
    end

    graph = scatter(P, ES, title="Entanglement Entropy vs\n log(density of occupied states) \n ",
                 label="N = $N and \nbipartition site $b", legendposition=:bottomright) # yerror = ESeb,
    xlabel!("log(Density of state vector) / log('p')\n ")
    ylabel!(" \nEntanglement Entropy\n ")

    maxS = round(b * log(2) - 4^b/(2^(N+1)), digits = 5)
    Y = maxS * ones(length(P)) 
    plot!(P, Y, label="\nMaximal expectation \nvalue of log(EE) = $maxS", ls=:dash, lc="red")
    display(graph)
end

#pvsEEplot(N, b)

#println(length(RandArr(N, 0.8)))
#rng = MersenneTwister(1234); D = 2^N
#length(randsubseq(rng, 1:D, 0.8))

# Below is code for computing the entanglement entropy of an eigenvector of 
# a random matrix.

function droplowerhalf(A::SparseMatrixCSC)
    m,n = size(A)
    rows = rowvals(A)
    vals = nonzeros(A)
    V = Vector{eltype(A)}()
    I = Vector{Int}()
    J = Vector{Int}()
    for i=1:n
        for j in nzrange(A,i)
            rows[j]>i && break
            push!(I,rows[j])
            push!(J,i)
            push!(V,vals[j])
        end
    end
    return sparse(I,J,V,m,n)
end

function Hrand(D, ρ)
    #=
    Constructs a random symmetric SparseArray which represents the Hamiltonian of a density ρ.

    From this, eigenvectors could in principle be found; 
    to save memory it's returned in a SparseArray but the diagsparse function
    could calculate the eigenvectors for such an object. 
    =#
    S = sprand(D, D, ρ)
    H = Symmetric(droplowerhalf(S))

    return H
end

function eigenvec(D, ρ, E, n)
    #=
    Returns the ith eigenvector out of a range of eigenvectors centred on 
    an energy E
    =#
    H = Hrand(D, ρ)
    return diagsparse(H, E, n)[1]#[:,i]
end

MV = eigenvec(2^N, p, E, 31); X = 1:30; Sarr = []

# Eigenvectors are normalised, which is good to know
#println(abs(1.0 - v' * v))

for i in X
    vi = MV[:,i]
    Si = EntEnt2(vi, b, N)
    push!(Sarr, Si)
end
y2 = round(b * log(2) - 4^b/(2^(N+1)), digits = 5)
t = "Entanglement entropy vs\n eigenvector labels closest to energy $E"
xl = "Eigenvector label closest to energy $E"; yl = "Entanglement entropy"
l1 = "Entanglement entropy of eigenvectors\n for N = $N and bipartition site $b"
l2 = "Maximum expectation value of log(EE) = $y2"

PlotXY(X, Sarr, y2, t, xl, yl, l1, l2)
