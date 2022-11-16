using Statistics

function σsq(Y, N, p)
    #=
    The function σsq computes the variance for an array y which are the expectation values of an operator, 
    for a system of N sites and a step which is an integer which defines the size of the energy window.

    A larger integer for step reduces the energy window size and loops over more windows so is more computationally
    demanding however should be more accurate as the local mean being used should be more accurate.

    Y is a vector of the expectation values in order of ascending energies, N is the system size and
    p is a parameter which determines the size of the energy window 2^(N-p) and the number of energy
    windows 2^p such that σsqtot is the sum of local variances divided by the number of energy windows.
    =#
    σsqv = []
    for j in 1:2^p
        yloc = Y[((j-1)*2^(N-p)+1) : j * 2^(N-p)]
        push!(σsqv, cov(yloc; corrected = false))
    end
    
    σsqtot = sum(σsqv) * 1/2^p
    #println("Variance for N = ", N, " is ", σsqtot)
    return σsqtot
end