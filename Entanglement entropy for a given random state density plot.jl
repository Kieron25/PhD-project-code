# Generating a plot of random state density vs entanglement entropy 
# of the random state of a given density p.

include("statetolabel.jl")
include("Calculation of the entanglement entropy.jl")

using Plots
using Statistics
using SparseArrays
using Random

# System size is fixed
N = 12; b = 6#; p = 0.05
#D = 2^N

function RandArr(N, p)
    A = Int64[]
    L = Int64(round((p) * 2^N))

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
    Pa = collect((1/D):(10/D):(0.12))
    Pb = collect(0.15:0.05:0.95)
    P = append!(Pa, Pb)

    ES = []; #ESeb = [] # Error for Entanglement entropy (S) and its error bars

    for p in P
        val = pvsEEpoint(p, D, N, b)
        esl = val#[1]
        #esebl = sqrt(val[2])
        append!(ES, esl)
        #append!(ESeb, esebl)
    end

    graph = plot(log.(P), log.(ES), title="log(Entanglement Entropy) vs\n log(density of occupied states) \n ",
                 label="N = $N and \nbipartition site $b", legendposition=:bottomright) # yerror = ESeb,
    xlabel!("log(Density of state vector) / log('p')\n ")
    ylabel!(" \nlog(Entanglement Entropy)\n ")

    maxS = round(log(b * log(2) - 4^b/(2^(N+1))), digits = 5)
    Y = maxS * ones(length(P))
    plot!(log.(P), Y, label=" Maximal expectation \nvalue of log(EE) = $maxS", ls=:dash, lc="red")
    display(graph)
end

pvsEEplot(N, b)

#println(length(RandArr(N, 0.8)))
#rng = MersenneTwister(1234); D = 2^N
#length(randsubseq(rng, 1:D, 0.8))