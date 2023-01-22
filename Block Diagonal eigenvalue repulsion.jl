using Random
using LinearAlgebra
using Plots
using SparseArrays
using Arpack
using BlockDiagonals

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

function BlockDiagM(D, n, ρ)
    DT = n * D
    neigv = DT - 1

    m = [Hrand(D, ρ)]
    for i in 2:n
        a = Hrand(D, ρ)
        append!(m, a)
    end

    M = BlockDiagonal(m)

    EVM = eigs(M, nev = neigv, which=:SR)[1]

    Δ = [EVM[2] - EVM[1]]; R = []
    for n in 2:lastindex(evλ)-1
        δ_n = EVM[n+1] - EVM[n]
        r_n = min(δ_n, Δ[n-1])/max(δ_n, Δ[n-1])
        append!(Δ, δ_n)
        append!(R, r_n)
    end

    bins = range(0, 1, length=80)
    graph = histogram(R, bins=bins, normalize=:pdf, label="Data from $DT x $DT \nrandom $n Block Diagonal matrix")
    xlabel!("r / min(δ_n, δ_n-1) ÷ max(δ_n, δ_n-1)")
    display(graph)
end

#BlockDiagM(200, 6, 0.8)


#D = 200; n = 6;
#D = 240; n = 5
#D = 300; n = 4
D = 10; n = 150
DT = n*D
ρ = 0.8; neigv = DT - 1
#A = Hrand(D, ρ); B = Hrand(D, ρ); C = Hrand(D, ρ);
#D1 = Hrand(D, ρ); E = Hrand(D, ρ); F = Hrand(D, ρ);
#G = Hrand(D, ρ); H = Hrand(D, ρ); J = Hrand(D, ρ);
#K = Hrand(D, ρ); L = Hrand(D, ρ); M1 = Hrand(D, ρ);
# Construct Block Diagonal Array
M = BlockDiagonal([Hrand(D, ρ) for j in 1:n])#, G, H, J, K, L, M1])

evλ = eigs(M, nev = neigv, which=:SR)[1]

Δ = [evλ[2] - evλ[1]]; R = []
for n in 2:lastindex(evλ)-1
    δ_n = evλ[n+1] - evλ[n]
    r_n = min(δ_n, Δ[n-1])/max(δ_n, Δ[n-1])
    append!(Δ, δ_n)
    append!(R, r_n)
end

l = 84
#display(Δ)
bins = range(1e-4, 1, length=l)
graph = histogram(R, bins=bins, normalize=:probability, label="Data from $DT x $DT matrix \nwith $n random Block Diagonal ")
xlabel!("r / min(δ_n, δ_n-1) ÷ max(δ_n, δ_n-1)")
display(graph)
