function ZZop2(Ψ, i, N)
    #=
    This utilises the fact that ZZ is returns +1 for parallel spins and -1 for antiparallel spins.

    mod1() is used to utilise periodic boundary conditions: N + 1 = 1 or mod1(N+1, N) = 1
    =#
    
    return Zop2(Ψ, mod1(i, N)) * Zop2(Ψ, mod1(i+1, N))
end