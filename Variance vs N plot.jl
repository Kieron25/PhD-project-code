using Plots # This package makes the graphs.
using LaTeXStrings # This package makes the LaTeX formatting for the axis labels.

include("H2.jl")
include("diag.jl")
include("expvals.jl")
include("variance.jl") # For σsq() function

function σsqvsN(J, h, g, max)
    #=
    This function plots the variation in σ^2 vs N
    =#
    s2 = []
    for N in 8:max
        M = diag(H2(N, J, h, g))
        y = expvals(M, 3, N)[2]
        push!(s2, σsq(y, N, 4))
    end

    plot([8:max], s2, legend=false)
    scatter!([8:max], s2, legend=false)
    xlabel!("\n "*L"N"*" \n ")
    ylabel!("\n "*L"σ(N)^2")
end

J = 1 ; h = (sqrt(5)+1)/4 ; g = (sqrt(5)+5)/8 #+ 0.1im
max = 12
σsqvsN(J, h, g, max)