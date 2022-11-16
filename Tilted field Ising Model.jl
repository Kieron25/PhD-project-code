#=
This code converts a classical state to an integer and back.
An emphasis is placed on classical spin states as superpositions of qubits aren't considered in this simple example.
A caveat is that I'm following the binary convention where the rightmost digit corresponds to 2^0 rather than 2^n, where n is the length of the string; 
the least significant digit is on the right of the bit string.
To save computational time later on, it may be worth removing the reverse() statements which may change the form of the Hamiltonian but it'll be 
equivalent to the previous Hamiltonian just the order of basis states will be different.

See later notes on the operators defined for the Hamiltonian, etc.
=# 

#ψ = [1.0; 0.0; 1.0] 
#s = length(ψ) # Could also define it as an integer

using LinearAlgebra # This package is imported so the Hamiltonian can be diagonalised and eigenvalues and eigenvectors can be obtained.
using Plots # This package makes the graphs.
using LaTeXStrings # This package makes the LaTeX formatting for the axis labels.
using Statistics # This package is used to get the variance, the size of the fluctuations on the graphs.

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

function statetolabel(Ψ)
    # Defined such that all 0 state is labelled as 1.
    ϕ = reverse(Ψ)
    L = sum(ϕ[i] * 2^(i-1) for i in 1:length(Ψ)) + 1
    #println(N)

    return Int64(L)
end

#labeltostate(4, s)
#statetolabel(ψ)
#println(statetolabel(ψ) == 4.0)

# The necessary operators needed in the Transverse Tilted Field Ising Model: X, Z and ZZ respectively.

function Xop(Ψ, i)
    #=
    a is the state of a single spin: if a = 1, the output is zero and if a = 0 the output is one.
    replace() function doesn't work so I take the i-1 elements of the vector, replace the ith and 
    append the remaining spin states to return the state with the X operator acting on the ith spin.
    =#
    dif = length(Ψ) - i
    arr2 = last(Ψ, dif)
    arr = first(Ψ, i-1)
    if Ψ[i] == 1.0
        #replace!(Ψ, Ψ[i] => 0.0, count = 1)
        push!(arr, 0.0)
        append!(arr, arr2)

    else
        #replace!(Ψ, Ψ[i] => 1.0, count = 1)
        push!(arr, 1.0)
        append!(arr, arr2)

    end
    #println(arr)
    return arr
end

function Xop2(Ψ, i)
    # This is a simplified form of the X operator acting on site i of a Fock state vector Ψ 

    # An integer rather than float is returned for better precision/ easier computation.
    Ψc  =copy(Ψ)
    Ψc[i] = 1 - Ψ[i]
    return Ψc
end

function Zop(Ψ, i) 
    #=
    A single Z operator leaves Ψ[i] = 1 unchanged and Ψ[i] = 0 changes the phase by π radians (multplying Ψ by -1). 

    1 corresponds to [1; 0] (spin up) and 0 is [0: 1] (spin down), the definition isn't too important as long as it's consistent
    =#
    if Ψ[i] == 1.0
        return Ψ
    else
        return -1*Ψ
    end
end

function Zop2(Ψ, i)
    # This function, applicable for H2(N), returns 1 or -1 prefactor depending on whether Ψ[i] is 1 or 0 (up or down)
    return (2 * Ψ[i] - 1.0)
end

function ZZop(Ψ, i)
    #=
    This function returns the correct spin factor depending on whether 
    spin pairs are parallel or antiparralel.

    The second function is a simpler form of ZZop but with ZZop2 both forms are obsolete.
    =#
    if Ψ[i] == 1.0
        if Ψ[i+1] == 1.0
            return Ψ
        else
            return -1*Ψ
        end
    else Ψ[i] == 0.0
        if Ψ[i+1] == 1.0
            return -1*Ψ
        else
            return Ψ
        end
    end
    #=
    if Ψ[i] == Ψ[i+1]
        return Ψ
    else
        return -1*Ψ
    end=#
end

function ZZop2(Ψ, i, N)
    #=
    This utilises the fact that ZZ is returns +1 for parallel spins and -1 for antiparallel spins.

    mod1() is used to utilise periodic boundary conditions: N + 1 = 1 or mod1(N+1, N) = 1
    =#
    
    return Zop2(Ψ, mod1(i, N)) * Zop2(Ψ, mod1(i+1, N))
end
        
#Xop(ψ, 3)
#ZZop(ψ, 1)

#=
# This code printed true for all i in 1:15 for l = 4. NOTE it is i due to how the labelling of states is defined in the respective codes.
l = 4
max = l^2 - 1
for i in 1:max
    println(i, ": ", statetolabel(labeltostate(i, l)) == i)
end
=#

# Construction of the Hamiltonian. The Transverse Tilted Field Ising Model has 3 terms in it. 
# prefactors of terms in the Hamiltonian
# J is the term for ZZop, g for Xop and h for Zop


function Hmatrix(dim, g)
    #=
    This constructs an empty Hamiltonian of size (dim, dim) of whatever variable type g is. 
    =#
    H = zeros(typeof(g), dim, dim)

    return H
end

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

function HttfI(N, k, J , h, g)
    #=
    This is the Hamiltonian for the transverse tilted field Ising model, for a system of N spin 1/2 sites (or a system where each site has 2 internal d.o.f.).
    k is the ket vector row, the row of the matrix corresponding to the state or the label of the state in binary + 1.
    =#
    
    dim = 2^N
    #H = zeros(dim, dim)
    #H = Array{ComplexF64, 2}(undef, dim, dim)
    H = zeros(typeof(g), dim, dim)
    Ψ = labeltostate(k, N)

    for i in 1:N
        ϕ = Xop(Ψ, i)
        #println(Ψ, ϕ)
        j = statetolabel(ϕ)
        #println(k, j)

        # This if loop utilises the hermiticity of the Hamiltonian
        if k <= j
            H[k, j] += g
            H[j, k] += conj(g)

        else
            H[k, j] += 0.0
        end
    end

    for i in 1:N
        ϕ = Zop(Ψ, i)
        #println("Zop", Ψ, ϕ)
        j = statetolabel(abs.(ϕ))
        #println(k, j)
        inn = sum(Ψ' .* ϕ)
        if inn != 0.0
            locval1 = h * inn/abs(inn)
            #println(j,", ", k,", Zop ", locval1)
            H[j, k] += locval1
        else
            H[j, k] += 0.0
        end
    end

    max = N-1
    for i in 1:max
        ϕ = ZZop(Ψ, i)
        j = statetolabel(abs.(ϕ))
        inn = sum(Ψ' .* ϕ)
        if inn != 0.0
            locval2 = J * inn/abs(inn)
            #println(j,", ", k,", ZZop ",locval2, ", ",sum(Ψ' .* ϕ) , "\n")
            H[j, k] += locval2
        else
            H[j, k] += 0.0
        end
    end

    # For some reason, it wasn't correctly returning H_11, so this is computing manually and
    # divided by 2^s so each loop in H(s) contributes the correct total to the output of H(s)
    # As 0 is defined as [0; 1], spin down, the -ve sign is appropriate for Zop and all spins 
    # are in the same direction so there's no -ve sign for ZZop terms.
    # H[1, 1] += (-N * h + J * (N-1))/2^N

    # This was later removed to go outside of the loop rather than being computed 2^N times

    return H
end

#HttfI(ψ)

function H(N, J , h, g)
    #=
    N is the size of the system (the number of sites), so dim is the number of states the system can be in.
    =#

    dim = 2^N
    #H = zeros(dim, dim)
    #H = Array{ComplexF64, 2}(undef, dim, dim)
    H = Hmatrix(dim, g)
    H[1, 1] += (-N * h + J * (N-1))

    for m in 1:dim
        #println(m)
        H += HttfI(N, m, J , h, g)
    end
    
    # This block of code sets small off diagonal elements to zero. I could probably improve this by eliminating small 
    # imaginary parts however this doesn't feel too important yet.
    for i in 1:dim
        for j in 1:dim
            H[i, j] = zerocorrect(H[i, j])
        end
    end
    #display(H)
    return H
end

#H(6)

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

N = 8; J = 1 ; h = (sqrt(5)+1)/4 ; g = (sqrt(5)+5)/8 #+ 0.1im
# N = 14 is the limit due to the memory availability of the PC; in theory it
# could run for larger systems on a computer with more memory or the use of 
# sparse matrices to reduce the memory requirements of the code.

#H2(N, J, h, g) 
#println(isapprox(H(N, J, h, g), H2(N, J, h, g)))

#H(5, J, h, g)
#H2(5, J, h, g)
#=
for n in 10:15
    #println("\nH(",n,"):")
    #@time H(n, J, h, g)
    #push!(N1, @time H(n, J, h, g))
    println("\nH2(",n,"):")
    @time H2(n, J, h, g)
    #push!(N2, @time H2(n, J, h, g))
    #println("For N = ", n, " it's ", isapprox(H(n, J, h, g), H2(n, J, h, g)), " that H(",n,") and H2(",n,") match.")
end =#

#=
There's a small rounding error between some diagonal elements but I believe that is due to the code, 
however both matrices are approximately equal which suggests it is a rounding error in calculations. 
A comparison of runtimes may be interesting to carry out.

H2(N) is as expected much faster than H(N), and can handle up to N = 15 before the memory is exceeded 
- this is a technical limitation so a computer with more memory could run the code for larger systems.
H2(N) becomes faster around N = 7 and signficantly faster at N = 10
=#

function diag(M)
    #=
     M is a matrix (the Hamiltonian)
     E are the eigenvalues, energies when M is a Hamiltonian and 
     vecs is an array of eigenvectors where vecs[:, n] returns the nth eigenvector.
     This array is P such that the diagonal form of M is Mdiag = inv(vecs)*M*vecs

     The function returns the eigenvalues, eigenvectors and the diagonalised form of M (the matrix M is in the eigenbasis)
    =#
    eig = eigen(M)
    E, vecs = eig.values, eig.vectors; 
    #=
    Mdiag = inv(vecs) * M * vecs
    max = Int64(sqrt(length(M)))
    for i in 1:max
        for j in 1:max
            Mdiag[i, j] = zerocorrect(Mdiag[i, j])
        end
    end=#
    
    # Quicker way of generating a diagonal hamiltonian.
    L = length(E)
    Mdiag = zeros(L, L)
    for r in 1:L
        Mdiag[r, r] += exp(E[r])
    end
    
    return Mdiag, vecs #, E
end

#Vec = diag(M1)[2][: , 2]

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

function expectZZ(v, i, N)
    #=
    This takes an eigenvector v and calculates the expectation value for Zop2 acting on the ith site out of N sites.
    This works as the eigenvector is a vector of coefficients which are weights on the Fock States which Zop2 acts on
    
    The function returns the expectation value of Zop2 acting on an eigenstate of the system.
    =#
    val = 0.0
    for m in eachindex(v)
        val += v[m] * conj(v[m]) * ZZop2(labeltostate(m, N), i, N)
    end

    return val
end

#expect(Vec, 1, N)

function expvals(Mat, i, N)
    #=
    This returns the data points to plot a graph of expectation values of an operator vs the energy density for a given eigenstate.

    β is a column of the matrix of Mat[2] which gives an eigenvector with the computational basis.

    The function returns x (Energies scaled by system size N) 
    and y (expectation values) data to reproduce figure 4.2
    =#

    Y = zeros(2^N)
    X = Mat[1]/ N
    #step = 2^Int64(round(N/5))
    for β in 1:2^N
        Y[β] += expectZ(Mat[2][:,β], i, N)
    end

    return X, Y
end

# With a plot of the expectation values vs energy eigenstates, I want to determine the size of the fluctuations on these graphs.
# To this end, I calculate the variance for small energy windows where the mean is approximately the same and sum the local variances 
# to get the total variance over all energies, as a measure of the fluctuation.

function σsq(Y, N, step)
    #=
    The function σsq computes the variance for an array y which are the expectation values of an operator, 
    for a system of N sites and a step which is an integer which defines the size of the energy window.

    A larger integer for step reduces the energy window size and loops over more windows so is more computationally
    demanding however should be more accurate as the local mean being used should be more accurate.
    =#
    σsqv = []
    for j in 1:2^step
        yloc = Y[((j-1)*2^(N-step)+1) : j * 2^(N-step)]
        push!(σsqv, cov(yloc; corrected = false))
    end
    
    σsqtot = sum(σsqv) * 1/2^step
    #println("Variance for N = ", N, " is ", σsqtot)
    return σsqtot
end

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
M1 = H2(N, J, h, g)
M = diag(M1)
#x, y = expvals(M, 3, N) # M contains the eigenenergies and an array of eigenvectors. 
#σsq(y, N, 5)
#σsqvsN(J, h, g, 13)

#=
# Plot of expectation values vs eigenenergies scaled by N
graph = Plots.plot(x, y, xlims = (-0.5, 0.5), ylims = (-0.4, 0.4), markers = true, label = "N = "*string(N))
xlabel!("\n "*L"\frac{E_n}{N}"*" \n ")
ylabel!("\n "*L"⟨n|Z_3|n⟩")
display(graph) =#

#= #for step = 5 to plot variance vs N
plot([8:14], [0.068365, 0.028124, 0.008378, 0.004029, 0.001068, 0.000409, 0.000124], legend=false)
scatter!([8:14], [0.068365, 0.028124, 0.008378, 0.004029, 0.001068, 0.000409, 0.000124], legend=false)
xlabel!("\n "*L"N"*" \n ")
ylabel!("\n "*L"σ(N)^2")=#

#= #for step = 5 and 1/2^step factor rather than 2^step/2^N to plot variance vs N
plot([8:13], [0.017091, 0.014062, 0.008378, 0.008058, 0.004271, 0.003272], legend=false)
scatter!([8:13], [0.017091, 0.014062, 0.008378, 0.008058, 0.004271, 0.003272], legend=false)
xlabel!("\n "*L"N"*" \n ")
ylabel!("\n "*L"σ(N)^2")=#


#= #for step = 7 to plot variance vs N with 2^step/2^N factor which is incorrect as sum has 
plot([8:14], [0.636707, 0.357521, 0.109656, 0.056595, 0.014870, 0.005953, 0.001639], legend=false)
scatter!([8:14], [0.636707, 0.357521, 0.109656, 0.056595, 0.014870, 0.005953, 0.001639], legend=false)
xlabel!("\n "*L"N"*" \n ")
ylabel!("\n "*L"σ(N)^2")=#

# The exponentiation of the Hamiltonian, both forms are quivalent but exp() will probably work quicker for larger N.

#exp(M1)
#display(M[2] * M[1] * inv(M[2]))