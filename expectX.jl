include("Xop2.jl")
include("labeltostate.jl")
include("statetolabel.jl")

function expectX(v, i, N)
    #=
    This takes an eigenvector v and calculates the expectation value for Xop2 acting on the ith site out of N sites.
    
    The function returns the expectation value of Xop2 acting on an eigenstate of the system.

    The commented out section is an earlier bit of code which produces the same thing but is computationally 
    more complicated so the latter approach is instead taken. This was for Zop2 but a similar thing could be
    written for Xop2

    statetolabel(Xop2(labeltostate(m, N), i)) finds the label for the state which overlaps with 
    Xop2(labeltostate(m, N), i), which returns the state X acting on v for each state in the 
    computational basis and val are the resulting weights of the the overlaps.
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
        n = statetolabel(Xop2(labeltostate(m, N), i))
        val += v[m] * conj(v[n])
    end

    return val
end