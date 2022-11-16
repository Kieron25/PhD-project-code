include("Zop2.jl")
include("labeltostate.jl")

function expectZ(v, i, N)
    #=
    This takes an eigenvector v and calculates the expectation value for Z2op acting on the ith site out of N sites.
    
    The function returns the expectation value of Zop2 acting on an eigenstate of the system.

    The commented out section is an earlier bit of code which produces the same thing but is computationally 
    more complicated so the latter approach is instead taken.
    =#
    #=
    vcop = copy(v)
    for m in 1:length(v)
        vcop[m] = vcop[m] * Zop2(labeltostate(m, N), i)
    end

    return v' * vcop
    =#
    val = 0.0
    for m in eachindex(v)
        val += v[m]* conj(v[m]) * Zop2(labeltostate(m, N), i)
    end

    return val
end