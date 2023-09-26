# Outline of Z operator on D states.

using SparseArrays
using Expokit

include("statetolabel.jl")
include("labeltostate.jl")
include("Domain Wall state.jl")
include("EastREM Hamiltonian.jl")
include("GOE EastREM Block IPR.jl")
include("Zop2.jl")

function ZDm(D, i)
    #=
    This function returns a DxD matrix for the Z operator
    acting on the i-th spin. This considers all states and
    converts them into the spin-vector basis of length N and
    operates on the spins as Zop2 does.

    D and N are closely related, so only D is supplied.
    =#
    I = collect(range(1, D)); V = Float64[] # It's important to define values are float for multiplying the matrix later on.
    N = Int64(log2(D + 1)) # plus 1 accounts for the excluded all spins down state.

    for j in I
        #append!(I, j)
        ψ = labeltostate(j+1, N)
        Zval = Zop2(ψ, i)
        append!(V, Zval)
    end
    M = sparse(I, I, V)
    return M
end

function Ut(H, t, Ψ)
    #=
    Use negative time for U†(t) going the other way in time.

    This returns the matrix exponent of the sparse Hamiltonian
    H evolved to a time t acting on an "initial" state Ψ.

    It may be worth defining time later on so the function only does the
    exponent of a matrix once and then this object is raised to a power t
    in each time step; this may prove more efficient.
    =#
    return expmv(-im*t, H, Ψ)
end

#ZDm(TotalD(7), 4)