#=
 This file aims to create a GOE chain matrix, 
 i.e. a matrix with equally sized random blocks on the diagonal
 and sparsely connecting nearest neighbour blocks on the off-diagonal, 
 and zero elsewhere
=#

using Random
using SparseArrays
using LinearAlgebra
using BlockArrays
using Arpack
using Plots
using StatsBase

#=
d, referred to sometimes as D, is the dimensionality of the individual blocks
Nb, sometimes called N, is the number of blocks along a row/ column

Γ, sometimes called J, is the hopping parameter - consistent with previous work

sp is a measure of the sparsity of the off-diagonal blocks. It gives the probability 
an element is non-zero which should work after averages over many matrices are considered.
It's equivalent to α in Christian's thesis and the literature
=#
d = 10; Nb = 10; D = Nb * d
Γ = 1; sp = 0.5

function M1(Nb, d, Γ, sp)
    #=
    This function creates a block matrix with dense diagonal blocks and 
    nearest off-diagonal blocks with a sparsity sp on average with elements Γ

    Each block is d x d, and the matrix is Nb x Nb blocks, so Nb x d x Nb x d in size
    
    This function only uses Random and LinearAlgebra as external packages
    =#
    
    rng = MersenneTwister();
    diagel = randn(rng, Float64, (d,d)) # diagonal elements - dense blocks
    J = Γ * Array(sprand(Bool, d, d, sp))
    
    H = [zeros(Float64, d, d) for i in 1:Nb, j in 1:Nb]
    H[diagind(H)] .= [diagel]
    H[diagind(H, -1)] .= [J]
    H[diagind(H, 1)] .= [J']
    HM = reduce(vcat, [reduce(hcat, H[i, :]) for i in 1:Nb])
    
    #display(HM)
    return HM
end

function M2(Nb, d, Γ, sp)
    #=
    This function creates a block matrix with dense diagonal blocks and 
    nearest off-diagonal blocks with a sparsity sp on average with elements Γ
    sp is analogous to α, the mean density of off-diagonal blocks...

    Each block is d x d, and the matrix is Nb x Nb blocks, so Nb x d x Nb x d in size
    
    Unlike in M1, each block is different (closer to what is desired). 

    This function returns a block matrix is a sparse matrix format so it's compatible with
    other code that utilises the SparseArrays module.
    =#
    D = Nb * d
    # L gives the size of each block dimension
    L = [d]
    for j in 2:Nb
        append!(L, d)
    end

    A = BlockArray(spzeros(Float64, D, D), L, L)
    rng = MersenneTwister(rand(1:10000));
    diagel = randn(rng, Float64, (d,d)) 
    A[Block(1,1)] = Symmetric(diagel, :U)

    #J = Γ * Array(sprand(Bool, d, d, sp))
    for j in 2:Nb
        a = j - 1
        J = Γ * Array(sprand(Bool, d, d, sp))
        A[Block(a,j)] = J'
        A[Block(j,a)] = J

        rngloc = MersenneTwister(rand(1:10000));
        diagelloc = randn(rngloc, Float64, (d,d)) 
        A[Block(j,j)] = Symmetric(diagelloc, :U)
    end
    As = Array(A) # returns the matrix in a sparse form
    #display(As)
    return As
end
## codecell ==========================================
#heatmap(Matrix(M2(16, 10, Γ, sp)))
## codecell ==========================================

function GOEEastREMM(Nb, Γ)
    #=
    This function creates a block matrix with dense diagonal blocks and 
    nearest off-diagonal blocks with a sparsity sp on average with elements Γ
    sp is analogous to α, the mean density of off-diagonal blocks...

    The matrix has an EastREM structure, so the k-th diagonal block has dimensions 2^k x 2^k
    This means off-diagonal blocks are rectangles of dimensions 2^k x 2^k+/-1

    This function returns a block matrix is a sparse matrix format so it's compatible with
    other code that utilises the SparseArrays module.

    =#
    # L gives the size of each block dimension
    L = [1]
    for j in 1:Nb-1 
        append!(L, 2^j)
    end
    D = sum(L)

    A = BlockArray(spzeros(Float64, D, D), L, L)
    rng = MersenneTwister(rand(1:10000));
    diagel = randn(rng, Float64, (1,1)) 
    A[Block(1,1)] = Symmetric(diagel, :U)

    for j in 2:Nb
        h = j - 1
        α = 1/ L[j] # Or maybe it should be Nb - j counting the frozen spins?
        J = Γ * Array(sprand(Bool, L[h], L[j], α))
        #J = Γ * Array(sprand(Bool, d, d, sp))
        A[Block(h,j)] = J
        A[Block(j,h)] = J'

        rngloc = MersenneTwister(rand(1:10000));
        diagelloc = randn(rngloc, Float64, (L[j],L[j])) 
        A[Block(j,j)] = Symmetric(diagelloc, :U)
    end
    As = Array(A) # matrix is still returned in a sparse form I think
    #display(As)
    return As, D
end
## codecell ==========================================
#println(typeof(GOEEastREMM(7, Γ)[1]))
## codecell ==========================================

function eigvals(M, D)
    #=
    A simple function which returns the eigenvalues in ascending order for an 
    arbitrary array and number of eigenvalues.
    =#
    ne = D - 1
    return eigs(M, nev = ne, which=:SR)[1]
end

function eigenlevstats(Nb, d, Γ, sp, Nm)
    #=
    The function fidns the level statistics and the average of this distribution of level spacings

    Nb is the number of blocks
    D is the dimensions of the blocks
    Γ is the hopping parameter in the matrix
    sp is the 

    Nm is the number of matrices you want to average over
    =#
    R = []; D = Nb * d
    for j in 1:Nm
        Mloc = M2(Nb, d, Γ, sp)
        evλ = eigvals(Mloc, D) # returns an array of eigenvalues in ascending order

        Δ = [evλ[2] - evλ[1]] # n = 1, Δ is now the array of differences which forms s when scaled by the mean
        for n in 2:lastindex(evλ)-1
            δ_nminus1 = Δ[n-1]
            δ_n = evλ[n+1] - evλ[n]
            append!(Δ, δ_n)
            r_n = min(δ_n, δ_nminus1)/max(δ_n, δ_nminus1)
            append!(R, r_n)
        end
    end
    #=
    # This code returns the level spacing distribution for the given set of parameters
    l = 84 ; #arbitrary value to set the number of bins
    Γp = round(Γ, 5) # A rounded value of Γ to make the plots look better
    binz = range(0, 1, length=l)
    graph = histogram(R, bins=binz, normalize=:pdf, label="Data from $Nm GOE chain matrices \nwith $Nb $d x $d blocks and Γ = $Γp")
    #xlabel!("s / difference in eigenvalues ÷ average difference")
    xlabel!("r / min(δ_n, δ_n-1) ÷ max(δ_n, δ_n-1)")
    display(graph)
    =#

    avR = sum(R)/length(R)
    #println("Average value of r_n for $Nb $d x $d block GOE chain matrix is $avR")
    return avR
end

function eigenlevstats2(Nb, Γ, sp, Nm)
    #=
    The function finds the level statistics and the average of this distribution of level spacings
    for a GOE EastREM matrix with Nb blocks; the dimensions of the blocks scale as 1, 2, 4, ..., 2^(Nb - 1)

    Nb is the number of blocks
    Γ is the hopping parameter in the matrix
    sp is the sparsity of the off-diagonal blocks

    Nm is the number of matrices you want to average over
    =#
    R = []
    for j in 1:Nm
        Mloc, D = GOEEastREMM(Nb, Γ)
        evλ = eigvals(Mloc, D) # returns an array of eigenvalues in ascending order

        Δ = [evλ[2] - evλ[1]] # n = 1, Δ is now the array of differences which forms s when scaled by the mean
        for n in 2:lastindex(evλ)-1
            δ_nminus1 = Δ[n-1]
            δ_n = evλ[n+1] - evλ[n]
            append!(Δ, δ_n)
            r_n = min(δ_n, δ_nminus1)/max(δ_n, δ_nminus1)
            append!(R, r_n)
        end
    end
    #=
    # This code returns the level spacing distribution for the given set of parameters
    l = 84 ; #arbitrary value to set the number of bins
    Γp = round(Γ, 5) # A rounded value of Γ to make the plots look better
    binz = range(0, 1, length=l)
    graph = histogram(R, bins=binz, normalize=:pdf, label="Data from $Nm GOE chain matrices \nwith $Nb $d x $d blocks and Γ = $Γp")
    #xlabel!("s / difference in eigenvalues ÷ average difference")
    xlabel!("r / min(δ_n, δ_n-1) ÷ max(δ_n, δ_n-1)")
    display(graph)
    =#

    avR = sum(R)/length(R)
    #println("Average value of r_n for $Nb $d x $d block GOE chain matrix is $avR")
    return avR
end

#eigenlevstats(Nb, d, Γ, sp, 700)

Γarr = -2:0.4:0.4

function Γarrplt(Γarr)
    Γarrplt = []
    for γ in Γarr
        γl = 10^γ
        append!(Γarrplt, γl)
    end
    return Γarrplt
end

function Jvsmean(Nb, d, sp, Γarr)
    #= 
    This function returns an array of mean values for different hopping parameters γ

    This works for any N, d, sp supplied so parameters can be varied and differences seen.
    =#
    μarr = []
    for γ in Γarr
        #println(γ)
        μ = eigenlevstats(Nb, d, γ, sp, 100) # averages over 100 disorder realisations (matrices)
        # For GOE Matrix
        #μ = eigenlevstats2(Nb, γ, sp, 100)
        append!(μarr, μ)
    end
    return μarr
end

function Jvsmean2(Narr, d, sp, Γ)
    #= 
    This function returns an array of mean values for different hopping parameters γ

    This works for any N, d, sp supplied so parameters can be varied and differences seen.
    =#
    μarr = []
    for n in Narr
        #println(γ)
        # For GOE chain Matrix - equal sized blocks
        #μ = eigenlevstats(n, d, Γ, sp, 100) # averages over 100 disorder realisations (matrices)
        # For GOE Matrix
        μ = eigenlevstats2(n, Γ, sp, 100)
        append!(μarr, μ)
    end
    return μarr
end

function meanvsΓ(NMin, Nstep, NMax, d, sp, Γarr)
    #=
    This function returns a plot of the mean value as a function of the hopping parameter for
    different N, to investigate how the parameter varies across a different number of blocks
    for different hopping strengths.
    =#
    v1 = NMin + Nstep; x = Γarrplt(Γarr)
    μNMin = Jvsmean(NMin, d, sp, x)

    graph = plot(x, μNMin, xscale=:log10, label="N = $NMin", markershape=:xcross, legendposition=:topright)
    xlabel!("Γ"); ylabel!("⟨r⟩") # The average level spacing of the quantity r
    ylims!(0.35, 0.55)
    #println(NMin)
    
    for N in v1:Nstep:NMax
        μN = Jvsmean(N, d, sp, x)
        plot!(x, μN, label="N = $N", markershape=:xcross, legendposition=:topleft)
        println(N)
    end

    display(graph)
end

function meanvsN(ΓMin, Γstep, ΓMax, d, sp, Narr)
    #=
    This function returns a plot of the mean value as a function of the hopping parameter for
    different N, to investigate how the parameter varies across a different number of blocks
    for different hopping strengths.
    =#
    v1 = ΓMin + Γstep;
    μNMin = Jvsmean2(Narr, d, sp, ΓMin)

    graph = plot(log.(Narr), log.(μNMin), label="Γ = $ΓMin", markershape=:xcross, legendposition=:outertopright)
    title!("ln⟨r⟩ vs ln(N) for GOE EastREM matrices")
    xlabel!("ln(N)"); ylabel!("ln⟨r⟩") # The average level spacing of the quantity r
    #ylims!(0.35, 0.55) #ylims!(log(0.37), log(0.54))
    println(ΓMin)
    
    for Γ in v1:Γstep:ΓMax
        μN = Jvsmean2(Narr, d, sp, Γ)
        plot!(log.(Narr), log.(μN), label="Γ = $Γ", markershape=:xcross, legendposition=:outertopright)
        println(Γ)
    end

    display(graph)
end
#=
## codecell ==========================================
meanvsΓ(4, 8, 28, d, sp, Γarr)
## codecell ==========================================
=#
## codecell ==========================================
#meanvsN(0.31, 0.10, 0.61, 10, 0.05, 6:1:11)
## codecell ==========================================