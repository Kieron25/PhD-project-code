include("labeltostate.jl")
include("statetolabel.jl")
include("Xop2.jl")
include("Zop2.jl")
include("ZZop2.jl")


function H2(N, J , h, g)
    #=
    This function was written following the recommendations of Achilleas Lazarides on 27/10/2022
    
    N is the number of spin sites, so dim is the number of different Fock states.

    N = 15 results in a noticable runtime however it does still run. This is an improvement upon H(N).
    =#
    dim = 2^N
    #H = zeros(dim, dim)
    #H = Array{ComplexF64, 2}(undef, dim, dim)
    H = zeros(typeof(g), dim, dim)

    for α in 1:dim
        Ψ = labeltostate(α, N)
        diagval = 0
        #max = N-1
        for j in 1:N #max
            ϕ = Xop2(Ψ, j)
            β = statetolabel(ϕ)
            if α < β
                H[α, β] += g
                H[β, α] += conj(g)
            else
                #H[α, β] += 0.0
            end
            diagval += h * Zop2(Ψ, j) # Effectively does Zop for spin up 1 -> 1+ve and spin down 0 -> -ve
            spinfac = ZZop2(Ψ, j, N) # This spin factor is +/- 1 depending on whether spins are parallel or not.
            diagval += J * spinfac
        end
        H[α, α] += diagval

        #=
        # Older code with open boundary conditions where Z_N*Z_1 wasn't counted 
        ϕ = Xop2(Ψ, N)
        β = statetolabel(ϕ)
        if α < β
            H[α, β] += g
            H[β, α] += conj(g)
        else
            H[α, β] += 0.0
        end
        diagval += h * Zop2(Ψ, N)
        H[α, α] += diagval
        =#
    end
    #display(H)
    return H
end

#H2(3, 1, 0.1, 0.2)