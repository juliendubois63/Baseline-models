using NLsolve
using Plots
using Random
using DataFrames
using ForwardDiff
using TimerOutputs

params = (; σ = 1.0,    # Coefficient of relative risk aversion
               β = 0.985,   # Discount factor
               δ = 0.02,   # Depreciation rate
               α = 0.2685,    # Capital share
               ϵ = 0.05,   # Elasticity of substitution
               Ae = 1.03,    # Energy saving TP
               A = 1.02,    # Capital TP
               R = 10)

# Create a new params with the updated β value
params = (; params..., β = params.β*(params.A^((1 - params.σ)/(1-params.α))), g = params.Ae/(params.A)^(1/(1-params.α)))

params.β*params.g

γ = 1e-2  # Threshold value for the extended functions

#if params.β*params.g > 1 we have g_k = g_c = g_y = 1 but g_e = (params.β*params.g)^ϵ
#if params.β*params.g = 1 we have g_k = g_c = g_e = g_y = 1
#if params.β*params.g < 1 we have g_k < g_c = g_y = g_e = (params.β*params.g)^(1/σ)

# Utility function and its derivative
function u(c::Float64, σ::Float64)
    return isone(σ) ? log(c) : (c^(1-σ)-1)/(1-σ)
end

function u_prime(c::Float64, σ::Float64)
    return isone(σ) ? 1/c : c^(-σ)
end

# Original production function (generic version)
function f(k::T, e::T, α::T, ϵ::T) where T<:Real
    return ((k^α)^((ϵ-1)/ϵ) + e^((ϵ-1)/ϵ))^(ϵ/(ϵ-1))
end

# Partial derivative with respect to e (generic version)
function f_prime_e(k::T, e::T, α::T, ϵ::T) where T<:Real
    return e^(-1/ϵ) * ((k^α)^((ϵ-1)/ϵ) + e^((ϵ-1)/ϵ))^(1/(ϵ-1))
end

# Partial derivative with respect to k (generic version)
function f_prime_k(k::T, e::T, α::T, ϵ::T) where T<:Real
    return α * (k^α)^(-1/ϵ) * ((k^α)^((ϵ-1)/ϵ) + e^((ϵ-1)/ϵ))^(1/(ϵ-1))
end

# Second derivatives for Taylor expansion
#function f_double_prime_e(k::Float64, e::Float64, α::Float64, ϵ::Float64)
##    return f_prime_e(k, e, α, ϵ)*(1/ϵ)*(e^(-1/ϵ)*((k^α)^((ϵ-1)/ϵ) + e^((ϵ-1)/ϵ))^(-1) - e^(-1)) 
#end

#function f_double_prime_k(k::Float64, e::Float64, α::Float64, ϵ::Float64)
#   return f_prime_k(k, e, α, ϵ)*(1/ϵ)*((α - 1)*k^(-1) - (k^α)^(-1) + α*k^(α - 1)*((k^α)^(-1/ϵ))*((k^α)^((ϵ-1)/ϵ) + e^((ϵ-1)/ϵ))^(-1))
#end

#Extended production function
function f_extension(k::Float64, e::Float64, α::Float64, ϵ::Float64, γ::Float64)
    if e > γ && k > γ
        return f(k, e, α, ϵ)
    elseif e <= γ && k > γ
        f_val = f(γ, γ, α, ϵ)
        δ = 1e-6
        grad_f = [(f(γ + δ, γ, α, ϵ) - f(γ, γ, α, ϵ)) / δ, (f(γ, γ + δ, α, ϵ) - f(γ, γ, α, ϵ)) / δ]
        hess_f = [(f(γ + δ, γ, α, ϵ) - 2 * f(γ, γ, α, ϵ) + f(γ - δ, γ, α, ϵ)) / δ^2,
                  (f(γ + δ, γ + δ, α, ϵ) - f(γ + δ, γ, α, ϵ) - f(γ, γ + δ, α, ϵ) + f(γ, γ, α, ϵ)) / δ^2,
                  (f(γ, γ + δ, α, ϵ) - 2 * f(γ, γ, α, ϵ) + f(γ, γ - δ, α, ϵ)) / δ^2]
        Δk = k - γ
        Δe = e - γ
        return f_val + grad_f[1] * Δk + grad_f[2] * Δe + 0.5 * (hess_f[1] * Δk^2 + 2 * hess_f[2] * Δk * Δe + hess_f[3] * Δe^2)
    elseif e > γ && k <= γ
        f_val = f(γ, γ, α, ϵ)
        δ = 1e-6
        grad_f = [(f(γ + δ, γ, α, ϵ) - f(γ, γ, α, ϵ)) / δ, (f(γ, γ + δ, α, ϵ) - f(γ, γ, α, ϵ)) / δ]
        hess_f = [(f(γ + δ, γ, α, ϵ) - 2 * f(γ, γ, α, ϵ) + f(γ - δ, γ, α, ϵ)) / δ^2,
                  (f(γ + δ, γ + δ, α, ϵ) - f(γ + δ, γ, α, ϵ) - f(γ, γ + δ, α, ϵ) + f(γ, γ, α, ϵ)) / δ^2,
                  (f(γ, γ + δ, α, ϵ) - 2 * f(γ, γ, α, ϵ) + f(γ, γ - δ, α, ϵ)) / δ^2]
        Δk = k - γ
        Δe = e - γ
        return f_val + grad_f[1] * Δk + grad_f[2] * Δe + 0.5 * (hess_f[1] * Δk^2 + 2 * hess_f[2] * Δk * Δe + hess_f[3] * Δe^2)
    else
        f_val = f(γ, γ, α, ϵ)
        δ = 1e-6
        grad_f = [(f(γ + δ, γ, α, ϵ) - f(γ, γ, α, ϵ)) / δ, (f(γ, γ + δ, α, ϵ) - f(γ, γ, α, ϵ)) / δ]
        hess_f = [(f(γ + δ, γ, α, ϵ) - 2 * f(γ, γ, α, ϵ) + f(γ - δ, γ, α, ϵ)) / δ^2,
                  (f(γ + δ, γ + δ, α, ϵ) - f(γ + δ, γ, α, ϵ) - f(γ, γ + δ, α, ϵ) + f(γ, γ, α, ϵ)) / δ^2,
                  (f(γ, γ + δ, α, ϵ) - 2 * f(γ, γ, α, ϵ) + f(γ, γ - δ, α, ϵ)) / δ^2]
        Δk = k - γ
        Δe = e - γ
        return f_val + grad_f[1] * Δk + grad_f[2] * Δe + 0.5 * (hess_f[1] * Δk^2 + 2 * hess_f[2] * Δk * Δe + hess_f[3] * Δe^2)
    end
end
# Extended partial derivative with respect to e
function f_prime_e_extension(k::Float64, e::Float64, α::Float64, ϵ::Float64, γ::Float64)
    if e > γ && k > γ
        return f_prime_e(k, e, α, ϵ)
    elseif e <= γ && k > γ
        f_prime_e_val = f_prime_e(γ, γ, α, ϵ)
        δ = 1e-6
        grad_f = [(f_prime_e(γ + δ, γ, α, ϵ) - f_prime_e(γ, γ, α, ϵ)) / δ, (f_prime_e(γ, γ + δ, α, ϵ) - f_prime_e(γ, γ, α, ϵ)) / δ]
        Δk = k - γ
        Δe = e - γ
        return f_prime_e_val + grad_f[1] * Δk + grad_f[2] * Δe
    elseif e > γ && k <= γ
        f_prime_e_val = f_prime_e(γ, γ, α, ϵ)
        δ = 1e-6
        grad_f = [(f_prime_e(γ + δ, γ, α, ϵ) - f_prime_e(γ, γ, α, ϵ)) / δ, (f_prime_e(γ, γ + δ, α, ϵ) - f_prime_e(γ, γ, α, ϵ)) / δ]
        Δk = k - γ
        Δe = e - γ
        return f_prime_e_val + grad_f[1] * Δk + grad_f[2] * Δe
    else
        f_prime_e_val = f_prime_e(γ, γ, α, ϵ)
        δ = 1e-6
        grad_f = [(f_prime_e(γ + δ, γ, α, ϵ) - f_prime_e(γ, γ, α, ϵ)) / δ, (f_prime_e(γ, γ + δ, α, ϵ) - f_prime_e(γ, γ, α, ϵ)) / δ]
        Δk = k - γ
        Δe = e - γ
        return f_prime_e_val + grad_f[1] * Δk + grad_f[2] * Δe
    end
end

# Extended partial derivative with respect to k
function f_prime_k_extension(k::Float64, e::Float64, α::Float64, ϵ::Float64, γ::Float64)
    if k > γ && e > γ
        return f_prime_k(k, e, α, ϵ)
    elseif k <= γ && e > γ
        f_prime_k_val = f_prime_k(γ, γ, α, ϵ)
        δ = 1e-6
        grad_f = [(f_prime_k(γ + δ, γ, α, ϵ) - f_prime_k(γ, γ, α, ϵ)) / δ, (f_prime_k(γ, γ + δ, α, ϵ) - f_prime_k(γ, γ, α, ϵ)) / δ]
        Δk = k - γ
        Δe = e - γ
        return f_prime_k_val + grad_f[1] * Δk + grad_f[2] * Δe
    elseif k > γ && e <= γ
        f_prime_k_val = f_prime_k(γ, γ, α, ϵ)
        δ = 1e-6
        grad_f = [(f_prime_k(γ + δ, γ, α, ϵ) - f_prime_k(γ, γ, α, ϵ)) / δ, (f_prime_k(γ, γ + δ, α, ϵ) - f_prime_k(γ, γ, α, ϵ)) / δ]
        Δk = k - γ
        Δe = e - γ
        return f_prime_k_val + grad_f[1] * Δk + grad_f[2] * Δe
    else
        f_prime_k_val = f_prime_k(γ, γ, α, ϵ)
        δ = 1e-6
        grad_f = [(f_prime_k(γ + δ, γ, α, ϵ) - f_prime_k(γ, γ, α, ϵ)) / δ, (f_prime_k(γ, γ + δ, α, ϵ) - f_prime_k(γ, γ, α, ϵ)) / δ]
        Δk = k - γ
        Δe = e - γ
        return f_prime_k_val + grad_f[1] * Δk + grad_f[2] * Δe
    end
end


# Define the function to compute next capital, consumption, and energy
function next(k::Float64, c::Float64, e::Float64, params::NamedTuple)

    # Compute next value of capital
    k_next = (f_extension(k, e, params.α, params.ϵ, γ) + (1 - params.δ) * k - c) * (1 / params.A^(1 - params.α))

    # System to compute c_next and e_next
    function equations!(E, x)
        c_next = x[1]
        e_next = x[2]
        
        E[1] = u_prime(c, params.σ) * params.A^(1 / (1 - params.α)) - params.β * u_prime(c_next, params.σ) * (f_prime_k_extension(k_next, e_next, params.α, params.ϵ, γ) + 1 - params.δ)
        E[2] = params.β * u_prime(c_next, params.σ) / u_prime(c, params.σ) * (f_prime_e_extension(k_next, e_next, params.α, params.ϵ, γ)/f_prime_e_extension(k, e, params.α, params.ϵ, γ)) - (1 / params.g)
    end
    
    # Solve for next consumption and efficiency using nlsolve
    x0 = [c, e]  # Initial guess for the solver
    sol = nlsolve(equations!, x0)
    c_next, e_next = sol.zero

    return k_next, c_next, e_next
end

function shooting(k0::Float64, c0::Float64, e0::Float64, params::NamedTuple, T::Int)
    c_path = [c0]
    k_path = [k0]
    e_path = [e0]
    for _ in 1:T
        c = c_path[end]
        k = k_path[end]
        e = e_path[end]
        
        k_next, c_next, e_next = next(k, c, e, params)
        push!(c_path, c_next)
        push!(k_path, k_next)
        push!(e_path, e_next)    
    end
    
    return c_path, k_path, e_path
end

function shooting_info(k0::Float64, c0::Float64, e0::Float64, params::NamedTuple, T::Int)
    # Initialize paths
    c_path, k_path, e_path = [c0], [k0], [e0]

    for t in 1:T
        c, k, e = c_path[end], k_path[end], e_path[end]

        try
            # Compute next step using next() function
            k_next, c_next, e_next = next(k, c, e, params)
            push!(c_path, c_next)
            push!(k_path, k_next)
            push!(e_path, e_next)
        catch ex
            # Return paths along with error details if an exception occurs
            return c_path, k_path, e_path, t, string(typeof(ex))
        end
    end
    return c_path, k_path, e_path, T, "No Error"
end

shooting_info(1.0, 1.0, 1.0, params, 10)

#This function tests for random values of k,c, and e when does the function next returns Domainerror
#One of these variables can be chosen as fixed by including it in the namedTuple
function test_shooting_errors_and_plot(params::NamedTuple, horizons::Vector{Int}, num_trials::Int, fixed_values::NamedTuple, sort_by::Symbol)
    results = DataFrame(T = Int[], k0 = Float64[], c0 = Float64[], e0 = Float64[], 
                        is_error = Bool[], error_type = String[], error_iteration = Union{Int, Missing}[], 
                        f_prime_e_val = Union{Float64, Missing}[], f_prime_k_val = Union{Float64, Missing}[])  # Use Union{Float64, Missing}

    for _ in 1:num_trials
        # Assign random values if not fixed, otherwise use fixed values
        k0 = get(fixed_values, :k, rand() * 3.0)
        c0 = get(fixed_values, :c, rand() * 3.0)
        e0 = get(fixed_values, :e, rand() * 3.0)

        for T in horizons
            c_path, k_path, e_path, error_iteration, error = shooting_info(k0, c0, e0, params, T)
            f_prime_e_val = round(f_prime_e_extension(k_path[end], e_path[end], params.α, params.ϵ, γ), digits=5)
            f_prime_k_val = round(f_prime_k_extension(k_path[end], e_path[end], params.α, params.ϵ, γ), digits=5)
            if error == "No Error"
                # Calculate f_prime_e_val and f_prime_k_val
                push!(results, (T, k0, c0, e0, false, "No Error", error_iteration, f_prime_e_val, f_prime_k_val))
            else
                push!(results, (T, k0, c0, e0, true, error, error_iteration, f_prime_e_val, f_prime_k_val))  
            end
        end
    end

    sort!(results, sort_by)
    println(results)

    # Unique horizons and plot setup
    unique_T = unique(results.T)
    num_T = length(unique_T)
    rows = cld(num_T, 2)
    layout = @layout [grid(rows, 2)]
    p = plot(layout = layout, size = (800, 400 * rows))

    for (i, T) in enumerate(unique_T)
        T_results = filter(row -> row.T == T, results)
        plot_title = "Error Occurrence for T=$T"

        # Extract data for error/no error cases
        c_no_error, c_error = T_results.c0[.!T_results.is_error], T_results.c0[T_results.is_error]
        k_no_error, k_error = T_results.k0[.!T_results.is_error], T_results.k0[T_results.is_error]
        e_no_error, e_error = T_results.e0[.!T_results.is_error], T_results.e0[T_results.is_error]

        if !haskey(fixed_values, :c) && haskey(fixed_values, :k) && haskey(fixed_values, :e)
            scatter!(c_no_error, zeros(length(c_no_error)), subplot = i, xlabel = "Consumption", ylabel = "Error", 
                     title = plot_title, label = "No Error", color = :blue)
            scatter!(c_error, ones(length(c_error)), subplot = i, label = "Error", color = :red)

        elseif haskey(fixed_values, :c) && !haskey(fixed_values, :k) && haskey(fixed_values, :e)
            scatter!(k_no_error, zeros(length(k_no_error)), subplot = i, xlabel = "Capital", ylabel = "Error", 
                     title = plot_title, label = "No Error", color = :blue)
            scatter!(k_error, ones(length(k_error)), subplot = i, label = "Error", color = :red)

        elseif haskey(fixed_values, :c) && haskey(fixed_values, :k) && !haskey(fixed_values, :e)
            scatter!(e_no_error, zeros(length(e_no_error)), subplot = i, xlabel = "Energy", ylabel = "Error", 
                     title = plot_title, label = "No Error", color = :blue)
            scatter!(e_error, ones(length(e_error)), subplot = i, label = "Error", color = :red)

        elseif !haskey(fixed_values, :c) && !haskey(fixed_values, :k) && haskey(fixed_values, :e)
            scatter!(c_no_error, k_no_error, subplot = i, xlabel = "Consumption", ylabel = "Capital", 
                     title = plot_title, label = "No Error", color = :blue)
            scatter!(c_error, k_error, subplot = i, label = "Error", color = :red)

        elseif !haskey(fixed_values, :c) && haskey(fixed_values, :k) && !haskey(fixed_values, :e)
            scatter!(c_no_error, e_no_error, subplot = i, xlabel = "Consumption", ylabel = "Energy", 
                     title = plot_title, label = "No Error", color = :blue)
            scatter!(c_error, e_error, subplot = i, label = "Error", color = :red)

        elseif haskey(fixed_values, :c) && !haskey(fixed_values, :k) && !haskey(fixed_values, :e)
            scatter!(k_no_error, e_no_error, subplot = i, xlabel = "Capital", ylabel = "Energy", 
                     title = plot_title, label = "No Error", color = :blue)
            scatter!(k_error, e_error, subplot = i, label = "Error", color = :red)

        else
            scatter!(c_no_error, k_no_error, e_no_error, subplot = i, xlabel = "Consumption", ylabel = "Capital", 
                     zlabel = "Energy", title = plot_title, label = "No Error", color = :blue)
            scatter!(c_error, k_error, e_error, subplot = i, label = "Error", color = :red)
        end
    end

    display(p)

    # Error Summary
    error_summary = combine(groupby(results, :error_type), nrow => :count)
    println("Summary of Error Types:")
    println(error_summary)
end


# Example usage
horizons = [10, 20, 25, 50]
num_trials = 100
fixed_values = (; k = 1.0)  # Put in the namedTuple the fixed values you want, others will be random

test_shooting_errors_and_plot(params, horizons, num_trials, fixed_values, :e0)

# Multivariate Newton-Raphson method
function norm(x::Vector{Float64})
    return sqrt(sum(xi^2 for xi in x))
end


function newton_raphson(k0::Float64, c0::Float64, e0::Float64, params::NamedTuple, T::Int; tol = 1e-6, max_iter = 1e3, λ = 0.2)
    β, σ, δ, α, A, R, g = params.β, params.σ, params.δ, params.α, params.A, params.R, params.g

    function f(x)
        c0, e0 = x
        result = try
            shooting(k0, c0, e0, params, T)
        catch ex
            if isa(ex, DomainError)
                return [Inf, Inf]
            else 
                rethrow(ex)
            end    
        end

        c_vec, k_vec, e_vec = result
        e_cons = sum(e_vec[i] / g^(i) for i in 1:T)
        k_ter = β^T * u_prime(c_vec[end], σ) * k_vec[end]
        return [e_cons - R, k_ter]
    end
    
    function jacobian(x)
        δ = 1e-6
        J = zeros(2, 2)
        
        for i in 1:2
            x_δ = copy(x)
            x_δ[i] += δ
            res_δ = f(x_δ)
            res = f(x)
            J[:, i] = (res_δ - res) / δ
        end
        
        return J
    end

    x = [c0, e0]  # Initial guess for c0 and e0
    last_valid_x = x
    rng = MersenneTwister()

    for i in 1:max_iter
        res = f(x)
        if any(isinf, res) || any(!isfinite, res)
            # Perturb previous valid guess with a small random noise
            x = last_valid_x .* (1 .+ 0.1 * (2 * rand(rng, 2) .- 1))  # 10% random perturbation
            continue
        end

        J = jacobian(x)
        Δx = -LinearAlgebra.pinv(J) * res
        x_new = x + Δx

        # Ensure c and e are positive
        x_new[1] = max(x_new[1], 1e-6)
        x_new[2] = max(x_new[2], 1e-6)

        # Apply damping factor
        x_new = λ * x_new + (1 - λ) * x

        # Check if the new x is valid
        try
            res_new = f(x_new)
            if any(isinf, res_new) || any(!isfinite, res_new)
                x_new = x .* (1 .+ 0.1 * (2 * rand(rng, 2) .- 1))  # 10% random perturbation
                continue
            end
            last_valid_x = x
            x = x_new
        catch ex
            if isa(ex, DomainError)
                Δx *= 0.5
                x_new = x + Δx
                x_new[1] = max(x_new[1], 1e-6)
                x_new[2] = max(x_new[2], 1e-6)
                x = x_new
                continue
            else
                rethrow(ex)
            end
        end

        if norm(Δx) < tol
            return x
        end
        if i % 25 == 0
            println("Iteration $i: c0 = $(x[1]), e0 = $(x[2]), res = $res")
        end
    end
    println("Newton-Raphson did not converge. Returning the last guess: c0 = $(x[1]), e0 = $(x[2])")
    return x
end

# Find the initial guess for c0 and e0
T = 50
k0 = 1.1 
c0 = 0.1
e0 = 0.2


#Compute the guesses
guess = newton_raphson(k0, c0, c0, params, T)
# Get the original paths
c_path, k_path, e_path = shooting(k0, guess[1], guess[2], params, T)


# Adjust the paths by multiplying consumption and capital by A^i and dividing efficiency by A^i
function adjust_paths(c_path, k_path, e_path, params)
    (; σ, β, δ, α, ϵ, Ae, A, R, g) = params
    T = length(c_path)
    c_adj = [c_path[i] * A^(i/(1-α)) for i in 1:T]
    k_adj = [k_path[i] * A^(i/(1-α)) for i in 1:T]
    e_adj = [e_path[i] / g^(i) for i in 1:T]
    return c_adj, k_adj, e_adj
end

# Adjust the paths
c_adj, k_adj, e_adj = adjust_paths(c_path, k_path, e_path, params)

# Function to plot the paths
function plot_paths(c_path, k_path, e_path, c_adj, k_adj, e_adj)
    plot(layout = (3, 2), size=(1000, 800))
    plot!(c_path, label="Consumption", lw=2, color="blue", xlabel="Time", ylabel="Consumption", title="Intensive Consumption Path", subplot=1, legend=:topright)
    plot!(k_path, label="Capital", lw=2, color="red", xlabel="Time", ylabel="Capital", title="Intensive Capital Path", subplot=3, legend=:topright)
    plot!(e_path, label="Energy", lw=2, color="green", xlabel="Time", ylabel="Energy", title="Intensive Energy Path", subplot=5, legend=:topright)
    plot!(c_adj, label="Adjusted Consumption", lw=2, color="blue", xlabel="Time", ylabel="Consumption", title="Adjusted Consumption Path", subplot=2, legend=:topright)
    plot!(k_adj, label="Adjusted Capital", lw=2, color="red", xlabel="Time", ylabel="Capital", title="Adjusted Capital Path", subplot=4, legend=:topright)
    plot!(e_adj, label="Adjusted Energy", lw=2, color="green", xlabel="Time", ylabel="Energy", title="Adjusted Energy Path", subplot=6, legend=:topright)
end

# Plot the paths
plot_paths(c_path, k_path, e_path, c_adj, k_adj, e_adj)

#This function looks for a feasible initial guess for c0 and e0 around the one obtained by the newton_raphson function
#It randomly tries for c0 and e0 values around the guess when newton_raphson has not converged
function find_feasible_initial_guess(guess::Vector{Float64}, params::NamedTuple, T::Int; tol = 0.1, max_iter = 10000)
    results = DataFrame(c0 = Float64[], e0 = Float64[], success = Bool[])
    Random.seed!(1234)  # For reproducibility

        for i in 1:max_iter
        c0 = max(guess[1] + (rand() - 0.5) * 2 * tol, 1e-6)
        e0 = max(guess[2] + (rand() - 0.5) * 2 * tol, 1e-6)
        try
            c_path, k_path, e_path = shooting(1.1, c0, e0, params, T)
            push!(results, (c0, e0, true))
            return c0, e0, results
        catch ex
            if isa(ex, DomainError)
                push!(results, (c0, e0, false))
            else
                rethrow(ex)
            end
        end
    end
    println("No feasible initial guess found within the iteration limit.")
    return nothing, nothing, results
end
# Example usage
c0_, e0_, results = find_feasible_initial_guess(guess, params, T)
# Get the original paths
c_path, k_path, e_path = shooting(k0, c0_, e0_, params, T)