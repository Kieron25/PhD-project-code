#using LinearAlgebra # For Dense Matrices
using Arpack # For Sparse Matrices

include("H2sparse.jl")

function diagsparse(M, E, nee)
    #=
     M is a matrix (the Hamiltonian)
     E is the energy you want eigenstates centred around,and nee is the number of eigenstates you want to calculate.
     vecs is an array of eigenvectors where vecs[:, n] returns the nth eigenvector.
     This array is P such that the diagonal form of M is Mdiag = inv(vecs)*M*vecs

     The function returns the eigenvalues, eigenvectors and the diagonalised form of M (the matrix M is in the eigenbasis)
    =#
    if nee == 0
        # This loop returns the eigenenergy and eigenvector for the lowest energy state
        eig0 = eigs(M, nev = 1, which = :SR)
        E0, vecs = eig0
        return vecs, E0
    else
        eig1 = eigs(M, nev = nee, maxiter = 40, sigma = E)
        E1, vecs1 = eig1
        return vecs1, E1
    end
    
    # Quicker way of generating a diagonal hamiltonian.
    #=
    L = length(E)
    Mdiag = zeros(L, L)
    for r in 1:L
        Mdiag[r, r] += exp(E[r])
    end
    =#
    # Don't need Mdiag, or could get it later from E. 
end

#=
N = 13; E = 0.01
H = H2sparse(N, 1, 0.21, 0.35)
diagsparse(H, E, 20)[1]
=#