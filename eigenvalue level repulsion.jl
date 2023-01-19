# Seeing eigenvalue level repulsion in random matrices 
# when a parameter λ is varied

using Arpack
using Plots
#using Random
using SparseArrays
using Statistics

#include("diagsparse.jl")

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

D = 400; 
ρ = 0.8; neigv = D-1 #λ = 0.1;
A = Hrand(D, ρ); B = Hrand(D, ρ); λ = 0.7
#=EV = []; X = 0.0:0.1:1
for λ in X
    Aλ = λ * A; Bλ = (1 - λ) * B
    Mλ = Aλ + Bλ
    evλ = eigs(Mλ, nev = neigv, which=:LR)[1]
    push!(EV, evλ)
end=#
#display(EV)
Aλ = λ * A; Bλ = (1 - λ) * B
Mλ = Aλ + Bλ
evλ = eigs(Mλ, nev = neigv, which=:SR)[1] # SR is given since we want eigenvalues in ascending order so pick the smallest ones first
#display(evλ)

Δ = []
for n in 1:lastindex(evλ)-1
    #δ_nminus1 = evλ[n] - evλ[n-1]
    δ_n = evλ[n+1] - evλ[n]
    #r_n = min(δ_n, δ_nminus1)/max(δ_n, δ_nminus1)
    append!(Δ, δ_n)
    #append!(Δ, r_n)
end
meanΔ = mean(Δ); l = 84
#display(Δ)
bins = range(1e-4, 5, length=l)
graph = histogram(Δ/meanΔ, bins=bins, normalize=:pdf, label="Data from $D x $D \nrandom matrix")
xlabel!("s / difference in eigenvalues ÷ average difference")
Y1 = []
for s in bins
    y = (pi/2) * s * exp( -(pi * s^ 2)/4)
    append!(Y1, y)
end
plot!(bins, Y1, label="Probability Distribution for \nGaussian Orthogonal Ensemble")
display(graph)
#l = D-1
#=
Y = []
for i in 1:lastindex(X)
    append!(Y, EV[i][1])
end
#display(Y)
graph1 = plot(X, Y, title="Eigenvalues for a $D x $D random symmetric matrix\n vs variable parameter λ\n ", label="eigenvalue 1",size = (800,450), legendposition=Symbol(:outer, :topright))
xlabel!("Variable parameter λ\n ")
ylabel!(" \nEigenvalues")

for i in 2:neigv
    Yj = []
    for j in 1:lastindex(X)
        append!(Yj, EV[j][i])
    end
    #display(Yj)
    plot!(X, Yj, label="eigenvalue $i")
end
display(graph1)
=#

#=
graph = scatter(X, Y, title=titlestr, label=label1, legendposition=:bottomright, ylims = (y2-0.15, y2+0.1)) # yerror = ESeb,
xlabel!(xlabelstr)
ylabel!(" \nEntanglement Entropy\n ")
    
y2arr = Y2 * ones(length(X)) 
plot!(X, y2arr, label=label2, ls=:dash, lc="red")
display(graph)
=#