include("ZZop2.jl")
include("labeltostate.jl")


function expectZZ(v, i, N)
    #=
    This takes an eigenvector v and calculates the expectation value for Zop2 acting on the ith site out of N sites.
    This works as the eigenvector is a vector of coefficients which are weights on the Fock States which Zop2 acts on
    
    The function returns the expectation value of Zop2 acting on an eigenstate of the system.
    =#
    val = 0.0
    for m in eachindex(v)
        val += v[m] * conj(v[m]) * ZZop2(labeltostate(m, N), i, N)
    end

    return val
end