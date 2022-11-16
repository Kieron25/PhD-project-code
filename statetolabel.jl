function statetolabel(Ψ)
    # Defined such that all 0 state is labelled as 1.
    ϕ = reverse(Ψ)
    L = sum(ϕ[i] * 2^(i-1) for i in 1:length(Ψ)) + 1
    #println(N)

    return Int64(L)
end