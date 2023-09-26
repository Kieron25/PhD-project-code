include("Domain Wall State.jl")
include("labeltostate.jl")
include("statetolabel.jl")
include("EastREM Hamiltonian.jl")
include("GOE EastREM Block IPR.jl")
include("Z operator on Fock space.jl")

using Statistics
using CurveFit
using JLD
#=
N = 8; Γ = 0.1; 
D = TotalD(N)
L = domainwalllabel(N)

Vspin = labeltostate(L, N)
# -1 is present in statevec since the all-down spin state is excluded from the GOE
# EastREM matrix as it's an isolated state from all the other states in the Fock Space
Vmat = statevec(L-1, D)
#println(Vmat, length(Vmat))

function time_evolve(ZD, ϕ, Ψf)
    # This function returns the product of time evolved vectors and a matrix ZD, for a single spin site
    # The past version of this function uses commented out U and Udag to evolve the states 
    # for each time step given as an argument in the function; this newly written function should work 
    # better
    val = Ψf * Array(ZD) * ϕ
    return Float64(real(val))
    #println(y)
    #println(typeof(y))
end

function func1!(ret, Nmin::Int64, Nmax::Int64, Ut::Matrix{ComplexF64}, Utdag::Adjoint{ComplexF64, Matrix{ComplexF64}}, ZD, ϕ, Ψf)
    #Nh = Nmax + 1 - Nmin
    for n in Nmin:Nmax
        i = n # For left hand side of the system; comment out for right side
        #i = n - Nh # This line is for the right side of the system; comment out for left side
        A = Ut * ϕ[i] ##
        B = Ψf[i] * Utdag ##
        val = time_evolve(ZD[i], A, B)
        ret[i] = val
        Ψf[i] = B; ϕ[i] = A ##
    end
    nothing
end

function ϕandZDdata(min::Int64, max::Int64, Dl::Int64, V::SparseVector{Int64, Int64})
    ϕArr = Any[]; ZDArr = Any[]; Ψfin = Any[]
    ϕf = V'
    for n in min:max
        ZD = ZDm(Dl, n)
        push!(ZDArr, ZD)
        ϕ = Array(ZD * V)
        push!(ϕArr, ϕ)
        push!(Ψfin, ϕf) ##
    end
    return ZDArr, ϕArr, Ψfin
end

function auto_corr(N::Int64, Γ::Float64, Nmin::Int64, Nmax::Int64, tstep::Int64, t::Int64)
    #=
    This function returns a value for 
    =#
    Ns = Nmax + 1 - Nmin; D = TotalD(N) 
    T = round(Int64, t/tstep); nit = 20
    ret = Vector{Float64}(undef, Ns)
    L = domainwalllabel(N); Y = zeros(Float64, T+1); Dl = TotalD(N)
    V = statevec(L-1, Dl)
    Y[1] = Float64(Ns * nit)
    ZDArr, ϕArr, Ψfin = ϕandZDdata(Nmin, Nmax, D, V)
    for m in 1:nit
        Ml = GOEEastREMM(N, Γ)[1]
        #println("Utl for $m")
        Utl = propagator(Matrix(Ml), tstep)
        Utldag = Utl'
        #Ψinit = ϕArr; Ψfin = V'
        for tl in 2:T+1 # tl = 1 corresponds to initial time t = 0
            func1!(ret, Nmin, Nmax, Utl, Utldag, ZDArr, ϕArr, Ψfin)
            Y[tl] += sum(ret)
        end
    end
    arr = Y/(Ns * nit)
    #display(arr)
    save("auto_corr$N and $Γ (L).jld", "Auto_corr_$N and $Γ (L)", arr) # and $Γ
    #(R) denotes right side of the system
end

function propagator(M, t)
    return exp(-im * M * t)
end

function avandstd(V)
    #=
    The mean and standard deviation of a vector V is calculated and used to plot a graph of
    the average vs 1/N with error bars plotted.
    =#
    μ = mean(V)
    err = std(V, mean = μ)
    return μ, err
end

function multiNplot(Lmin, Lmax, lr, tstep, tmax, Γ)
    #=
    This function is a crude way of plotting autocorrelation functions for the left/ right halves
    of spin systems of different sizes. 
    =#
    T = 0:tstep:tmax
    if lr == "L"
        println("L side selected.")
        auto_corr(Lmin, Γ, 1, Int64(Lmin/2), tstep, tmax)
        Y = load("auto_corr$Lmin and $Γ (L).jld")["Auto_corr_$Lmin and $Γ (L)"]
        #=graph = plot(T, Y, label="N = $Lmin", markershape=:circle, legendposition=:bottomright)
        title!("A_"*lr*" (t) for system sizes $Lmin to $Lmax\n and Hopping parameter Γ = $Γ\n ")
        xlabel!("time/ t"); ylabel!("A_"*lr*" (t)"); ylims!(0, 1.1)=#
        μ, err = avandstd(Y[4:length(Y)])
        #println(Lmin, " ", Γ,   " and the mean is ", μ, " \n ")
        Lminp1 = Lmin + 2
        X = [2/Lmin]; Y = [μ]; Yerr = [err]
        for L in Lminp1:2:Lmax
            auto_corr(L, Γ, 1, Int64(L/2), tstep, tmax)
            Yl = load("auto_corr$L and $Γ (L).jld")["Auto_corr_$L and $Γ (L)"]
            μl, errl = avandstd(Yl[4:length(Yl)])
            append!(X, 2/L); append!(Y, μl); append!(Yerr, errl)
            #plot!(graph, T, Yl, label="N = $L", markershape=:circle)
            #println(L, " and the mean is ", μl," \n ")
        end
        graph = scatter(X, Y, yerror=Yerr)
        title!("long-time average of A_"*lr*" (t) for system sizes $Lmin to $Lmax \n and Hopping parameter Γ = $Γ")
        xlabel!("1/N"); ylabel!("average A_"*lr*" (t)"); ylims!(0.0, 1.0)
        a, b = linear_fit(X, Y)
        Yf = a.*ones(length(X)) + b .* X
        plot!(X, Yf, label="Best fit line")
        println("intercept is $a and gradient is $b \n")
        display(graph)
    elseif lr == "R"# for the Right-hand side of the system
        println("R side selected.")
        Nmin = Int64(Lmin/2)+1
        auto_corr(Lmin, Γ, Nmin, Lmin, tstep, tmax)
        Y = load("auto_corr$Lmin and $Γ (R).jld")["Auto_corr_$L and $Γ (R)"]
        #=graph = plot(T, Y, label="N = $Lmin", markershape=:circle)
        title!("A_"*lr*" (t) for system sizes $Lmin to $Lmax\n and Hopping parameter Γ = $Γ")
        xlabel!("time/ t"); ylabel!("A_"*lr*" (t)"); ylims!(0, 1.1)=#
        μ, err = avandstd(Y[3:length(Y)])
        X = [2/Lmin]; Y = [μ]; Yerr = [err]
        Lminp1 = Lmin + 2
        #println(Lmin, " ", Γ, " and the mean is ", μ, " \n ")
        for L in Lminp1:2:Lmax
            auto_corr(L, Γ, Int64(L/2)+1, L, tstep, tmax)
            Yl = load("auto_corr$L and $Γ (R).jld")["Auto_corr_$L and $Γ (R)"]
            #plot!(graph, T, Yl, label="N = $L", markershape=:circle)
            μl, errl = avandstd(Yl[3:length(Yl)])
            append!(X, 2/L); append!(Y, μl); append!(Yerr, errl)
            #println(L,  " ", μl, " \n ")
        end
        graph = scatter(X, Y, yerror=Yerr)
        title!("long-time average of A_"*lr*" (t) for system sizes $Lmin to $Lmax \n and Hopping parameter Γ = $Γ")
        xlabel!("1/N"); ylabel!(" \naverage A_"*lr*" (t)"); ylims!(0.0, 1.0)
        a, b = linear_fit(X, Y)
        Yf = a.*ones(length(X)) + b .* X
        plot!(X, Yf, label="Best fit line")
        println("intercept is $a and gradient is $b \n ")
        display(graph)
    else
        println("Argument lr not supported; use capital L or R.")
    end
end

#Ute = propagator(Matrix(GOEEastREMM(10, 0.1)[1]), 10); V = statevec(domainwalllabel(10)-1, TotalD(10))
#println(typeof(Ute), " ", typeof(V))

#@time auto_corr(12, 0.1, 1, 6, 10, 50)
#@time func1!(Vector{Float64}(undef, 5), 10, 1, 5, Ute, V, 10)

#multiNplot(6, 12, "L", 10, 60, 0.1)
#multiNplot(6, 12, "L", 10, 60, 1.0)

#=
T = 0:10:60
auto_corr(6, 0.1, 1, 3, 10, 60)
Y = load("auto_corr6 and 0.1 (L).jld")["Auto_corr_6 and 0.1 (L)"]
graph = plot(T, Y, label="N = 6", markershape=:circle)
title!("A_L (t) for system sizes 6 to 12\n and Hopping parameter Γ = 0.1")
xlabel!("time/ t"); ylabel!("A_L (t)"); ylims!(0, 1.1)
for N in 8:2:12
    Yl = load("auto_corr$N and 0.1 (L).jld")["Auto_corr_$N and 0.1 (L)"]
    plot!(graph, T, Yl, label="N = $N", markershape=:circle)
end
display(graph)=#
