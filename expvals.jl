include("expectZ.jl")

function expvals(Mat, i, N)
    #=
    This returns the data points to plot a graph of expectation values of an operator vs the energy density for a given eigenstate.

    β is a column of the matrix of Mat[2] which gives an eigenvector with the computational basis.

    The function returns x (Energies scaled by system size N) 
    and y (expectation values) data to reproduce figure 4.2
    =#

    Y = zeros(2^N)
    X = Mat[1]/ N
    #step = 2^Int64(round(N/5))
    for β in 1:2^N
        Y[β] += expectZ(Mat[2][:,β], i, N)
    end

    return X, Y
end