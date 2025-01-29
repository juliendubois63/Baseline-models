using NLsolve
using Plots

# Define parameters as a named tuple for easy access
params = (; σ = 1.0,    # Coefficient of relative risk aversion
               β = 0.985,   # Discount factor
               δ = 0.02,   # Depreciation rate
               α = 0.2632,    # Capital share
               ϵ = 0.02,   # Elasticity of substitution
               Ae = 1.0,    # Energy saving TP
               A = 1.0,    # Capital TP
               R = 10)

# Production function and its derivatives
# Utility function and its derivative
u(c::Float64, σ::Float64) = isone(σ) ? log(c) : (c^(1-σ)-1)/(1-σ)
u_prime(c::Float64, σ::Float64) = isone(σ) ? 1/c : c^(-σ)

# Production function and its derivatives
function f(k::Float64, e::Float64, α::Float64, ϵ::Float64)
    return ((A*k^α)^((ϵ-1)/ϵ) + Ae*e^((ϵ-1)/ϵ))^(ϵ/(ϵ-1))
end

function f_prime_e(k::Float64, e::Float64, α::Float64, ϵ::Float64)
    return e^(-1/ϵ)*((k^α)^((ϵ-1)/ϵ) + e^((ϵ-1)/ϵ))^(1/(ϵ-1)) 
end

function f_prime_k(k::Float64, e::Float64, α::Float64, ϵ::Float64)
    return α*(k^α)^(-1/ϵ)*((k^α)^((ϵ-1)/ϵ) + e^((ϵ-1)/ϵ))^(1/(ϵ-1)) 
end

function next(k::Float64, c::Float64, e::Float64, params)
    # Unpack parameters
    β, δ, α, σ, ϵ, A, Ae = params.β, params.δ, params.α, params.σ, params.ϵ, params.A, params.Ae
    
    k_next = f(k, e, α, ϵ) + (1 - δ) * k - c

    function equations!(E, x)
        c_next = x[1]
        e_next = x[2]
        
        E[1] = u_prime(c, σ) - β * u_prime(c_next, σ) * (f_prime_k(k_next, e_next, α, ϵ) + (1 - δ))
        E[2] = β * u_prime(c_next, σ) / u_prime(c, σ) * f_prime_e(k_next, e_next, α, ϵ) / f_prime_e(k, e, α, ϵ) - 1
    end
    
    # Solve for next consumption and efficiency using nlsolve
    x0 = [c, e]
    sol = nlsolve(equations!, x0)
    c_next, e_next = sol.zero

    return k_next, c_next, e_next
end

function shooting(k0::Float64, c0::Float64, e0::Float64, T::Int, params)
    # Initialize arrays to store the paths
    k_path = [k0]
    c_path = [c0]
    e_path = [e0]
    
    # Iterate forward
    for i in 1:T
        c = c_path[end]
        k = k_path[end]
        e = e_path[end]
        
        k_next, c_next, e_next = next(k, c, e, params)

        push!(c_path, c_next)
        push!(k_path, k_next)
        push!(e_path, e_next)
    end
    return k_path, c_path, e_path
end

function energy_newton(k0::Float64, c0::Float64, params::NamedTuple, T::Int ; tol = 1e-6, max_iter = 1000)
    e = k0

    function f(e)
        k_path, c_path, e_path = shooting(k0, c0, e, T, params)
        try
            k_path, c_path, e_path = shooting(k0, c0, e, T, params)
        catch err
            if err isa DomainError
                e *= 1.1
                while true
                    try
                        k_path, c_path, e_path = shooting(k0, c0, e, T, params)
                        break
                    catch err
                        if err isa DomainError
                            e *= 2
                        else
                            rethrow(err)
                        end
                    end
                end
                return f(e)
            else
                rethrow(err)
            end
        end
        return sum(e_path) - params.R
    end

    # Differentiate f(e)
    function Df(e)
        h = 1e-6
        return (f(e + h) - f(e)) / h        
    end

    # Apply Newton's method
    for i in 1:max_iter
        e = e - f(e) / Df(e)
        if abs(f(e)) < tol
            println("Newton's method has converged after $i iterations.")
            return e
        end
        println("Iteration $i: e = $e, f(e) = $(f(e))")
    end
    
    println("Newton's method did not converge after $max_iter iterations.")
    return e
end

function bisection(k0::Float64, c0::Float64, params::NamedTuple, T::Int64 ; tol = 1e-6, max_iter::Int64 = 1000)
    e = energy_newton(k0, c0, params, T)
    c_upp = f(k0, e, params.α, params.ϵ) + (1 - params.δ) * k0
    c_low = 0.0
    c = c0
    i = 0

    while i < max_iter
        try
            k_path = shooting(k0, c, energy_newton(k0, c, params, T), T, params)[1]
        catch e
            if e isa DomainError
                c_upp = c
                c = (c_upp + c_low) / 2
                i += 1
                continue
            else
                rethrow(e)
            end
        end

        k_path = shooting(k0, c, energy_newton(k0, c, params, T), T, params)[1]
        error = k_path[end]

        if abs(error) < tol
            return c
        elseif error > 0
            c_low = c
        else
            c_upp = c    
        end
        c = (c_upp + c_low) / 2
        i += 1

        if i == max_iter
            println("Bisection method did not converge after $max_iter iterations.")
        end
    end
    return c
end

k0 = 1.1
c_opt = bisection(k0,0.1*k0,params,100)
e_opt = energy_newton(k0,c_opt,params,100)

k_path, c_path, e_path = shooting(1.1, c_opt, e_opt, 100, params)

function plot_all(k_path::Vector, c_path::Vector,e_path::Vector)
    p1 = plot(1:length(k_path), k_path, label = "Capital", title = "Capital Path", color = :blue)
    p2 = plot(1:length(c_path), c_path, label = "Consumption", title = "Consumption Path", color = :green)
    p3 = plot(1:length(e_path), e_path, label = "Energy", title = "Energy Path", color = :red)
    plot(p1, p2, p3, layout = (3, 1), legend = :topright)
end

plot_all(k_path, c_path, e_path)
sum(e_path)
e_grw = [e_path[i+1]/e_path[i] for i in 1:length(e_path)-1]