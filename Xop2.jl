function Xop2(Ψ, i)
    # This is a simplified form of the X operator acting on site i of a Fock state vector Ψ 

    # An integer rather than float is returned for better precision/ easier computation.
    Ψc  =copy(Ψ)
    Ψc[i] = 1 - Ψ[i]
    return Ψc
end