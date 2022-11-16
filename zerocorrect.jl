function zerocorrect(val)
    #= 
    An attempt to make a neater form of the Hamiltonian by setting all values with a small absolute mangitude to 0.0
    This isn't strictly necessary to include in plotting but it makes the Hamiltonian look neater.
    =#
    if abs(val) <= 1e-10
        val = 0.0
    else
        val = val
    end
    return val
end