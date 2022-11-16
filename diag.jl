using LinearAlgebra

include("H2.jl")

function diag(M)
    #=
     M is a matrix (the Hamiltonian)
     E are the eigenvalues, energies when M is a Hamiltonian and 
     vecs is an array of eigenvectors where vecs[:, n] returns the nth eigenvector.
     This array is P such that the diagonal form of M is Mdiag = inv(vecs)*M*vecs

     The function returns the eigenvalues, eigenvectors and the diagonalised form of M (the matrix M is in the eigenbasis)
    =#
    eig = eigen(M)
    E, vecs = eig.values, eig.vectors; 
    #println(size(vecs))
    #=
    Mdiag = inv(vecs) * M * vecs
    max = Int64(sqrt(length(M)))
    for i in 1:max
        for j in 1:max
            Mdiag[i, j] = zerocorrect(Mdiag[i, j])
        end
    end=#
    
    # Quicker way of generating a diagonal hamiltonian.
    L = length(E)
    Mdiag = zeros(L, L)
    for r in 1:L
        Mdiag[r, r] += exp(E[r])
    end
    
    return Mdiag, vecs, E
end
#=
N = 10
H = H2(N, 1, 0.21, 0.35)
diag(H)[2][:,100]
=#