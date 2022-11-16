function labeltostate(L, N)
    #=
    L is the label of the state and N is the size of the system.
    L-1 is defined such that state labelled 1 corresponds to all 0 state.
    So the binary value of the state corresponds to the value of the label - 1;
    state labelled 1 has all zeros, which would be 0...0 in binary
    =#
    val = digits(L-1, base = 2, pad = N)
    reverse!(val)
    #println(val)

    return val
end