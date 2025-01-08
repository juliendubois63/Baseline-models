using NLsolve
using Plots

# Define parameters as a named tuple for easy access
params = (; σ = 2.0,    # Coefficient of relative risk aversion
               β = 0.95,   # Discount factor
               δ = 0.02,   # Depreciation rate
               α = 0.33,    # Capital share
               A = 1.0,    # Total factor productivity
               R = 10)    

# Utility function and its derivative
u(c::Float64, σ::Float64) = isone(σ) ? log(c) : (c^(1-σ)-1)/(1-σ)
u_prime(c::Float64, σ::Float64) = isone(σ) ? 1/c : c^(-σ)
u_prime_inv(c::Float64, σ::Float64) = isone(σ) ? 1/c : c^(-1/σ)

# Production function and its derivatives
function f(k::Float64,e::Float64, α::Float64, A::Float64)
    return A * k^α * e^(1-α)
end

function f_prime_k(k::Float64,e::Float64, α::Float64, A::Float64)
    return α * A * k^(α-1) * e^(1-α)
    
end

function f_prime_e(k::Float64,e::Float64, α::Float64, A::Float64)
    return (1-α) * A * k^α * e^(-α)
end

# Define the function to compute next capital, consumption, and efficiency
function next_val(k::Float64, e::Float64, c::Float64, params::NamedTuple)
    α = params.α
    A = params.A
    δ = params.δ
    β = params.β
    σ = params.σ

    # Calculate next capital (state transition law)
    k_next = f(k, e, α, A) + (1 - δ) * k - c

    # Function to solve the system using the Euler equation
    function equations!(E, x)
        c_next = x[1]
        e_next = x[2]
        
        E[1] = u_prime(c, σ) - β * u_prime(c_next, σ) * (f_prime_k(k, e_next, α, A) + (1 - δ))
        E[2] = β * u_prime(c_next, σ) / u_prime(c, σ) * f_prime_e(k_next, e_next, α, A) / f_prime_e(k, e, α, A) - 1
    end
    
    # Solve for next consumption and efficiency using nlsolve
    x0 = [c, e]
    sol = nlsolve(equations!, x0)
    c_next, e_next = sol.zero

    return k_next, c_next, e_next
end

# Shooting method to compute consumption and capital paths for T periods
function shooting(k0::Float64, c0::Float64, e0::Float64, params::NamedTuple, T::Int)
    β = params.β
    σ = params.σ
    δ = params.δ
    α = params.α
    A = params.A
    R = params.R

    c_path = [c0]
    k_path = [k0]
    e_path = [e0]

    for i in 1:T
        c = c_path[end]
        k = k_path[end]
        e = e_path[end]
        
        k_next, c_next, e_next = next_val(k, e, c, params)

        push!(c_path, c_next)
        push!(k_path, k_next)
        push!(e_path, e_next)
    end

    return c_path, k_path, e_path
end

shooting(1.1, 0.1, 0.1, params, 50)

using Base.Threads: @async, sleep
function energy_optimal(k0::Float64, c0::Float64, e0::Float64, params::NamedTuple, T::Int ; tol = 1e-6, max_iter = 1000)
    β = params.β
    σ = params.σ
    δ = params.δ
    α = params.α
    A = params.A
    R = params.R
    i = 0
    e_upp = R
    e_low = 0.0
    e = e0

    timeout = 1.0  # Timeout in seconds
    start_time = time()
    elapsed_time = 0.0

    while i < max_iter
        elapsed_time = time() - start_time
        if elapsed_time > timeout
            println("Timeout: Function took longer than $timeout seconds at iteration $i.")
            return nothing
        end

        result = try
            shooting(k0, c0, e, params, T)
        catch ex
            if isa(ex, DomainError)
                e_upp = e
                e = (e_upp + e_low) / 2
                continue
            else
                rethrow(ex)
            end
        end

        if result === nothing
            e_upp = e
            e = (e_upp + e_low) / 2
            continue
        end

        c_vec, k_vec, e_vec = result
        e_cons = sum(e_vec)

        if e_cons > R
            e_upp = e
        else
            e_low = e
        end
        e = (e_upp + e_low) / 2
        i += 1
    end
    return e
end


# Initial values
function bisection(k0::Float64, c0::Float64, params::NamedTuple, T::Int ; tol = 1e-6, max_iter = 1000)
    c = c0
    c_upp = f(k0, energy_optimal(k0, c, 0.1, params, T), params.α, params.A) + (1 - params.δ) * k0
    c_low = 0.0
    
    i = 0

    timeout = 300.0  # Timeout in seconds (5 minutes)
    start_time = time()
    elapsed_time = 0.0

    while i < max_iter
        elapsed_time = time() - start_time
        if elapsed_time > timeout
            println("Timeout: Function took longer than $timeout seconds at iteration $i.")
            println("Current result: c = $c")
            return c
        end

        e_opt = energy_optimal(k0, c, 0.1, params, T)
        if e_opt === nothing
            c_upp = c
            c = (c_upp + c_low) / 2
            continue
        end

        e = e_opt

        result = try
            shooting(k0, c, e, params, T)
        catch ex
            if isa(ex, DomainError)
                c_upp = c
                c = (c_upp + c_low) / 2
                continue
            else
                rethrow(ex)
            end
        end

        if result === nothing
            c_upp = c
            c = (c_upp + c_low) / 2
            continue
        end

        c_vec, k_vec, e_vec = result
        error = u_prime(c_vec[end], params.σ) - params.β * u_prime(c_vec[end-1], params.σ) * (1 + f_prime_k(k_vec[end-1], e_vec[end-1], params.α, params.A) - params.δ)

        if abs(error) < tol
            return c
        elseif error > 0
            c_low = c
        else
            c_upp = c
        end
        c = (c_upp + c_low) / 2
        i += 1
    end
    return c
end


plot(shooting(1.1, bisection(1.1, 0.1, params, 100), 0.1, params, 100)[1], label="Consumption", lw=2, color="blue", xlabel="Time", ylabel="Consumption", title="Consumption Path")
plot(shooting(1.1, bisection(1.1, 0.1, params, 100), 0.1, params, 100)[2], label="Capital", lw=2, color="red", xlabel="Time", ylabel="Capital", title="Capital Path")
plot(shooting(1.1, bisection(1.1, 0.1, params, 100), 0.1, params, 100)[3], label="Efficiency", lw=2, color="green", xlabel="Time", ylabel="Efficiency", title="Efficiency Path")