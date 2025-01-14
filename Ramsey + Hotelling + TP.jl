using NLsolve
using Plots

params = (; σ = 1.0,    # Coefficient of relative risk aversion
               β = 0.95,   # Discount factor
               δ = 0.02,   # Depreciation rate
               α = 0.33,    # Capital share
               A = 1.09,    # Total factor productivity
               R = 10)             

# Create a new params with the updated β value
params = (; params..., β = params.β / (params.A^(1 - params.σ)))

# Utility function and its derivative
u(c::Float64, σ::Float64) = isone(σ) ? log(c) : (c^(1-σ)-1)/(1-σ)
u_prime(c::Float64, σ::Float64) = isone(σ) ? 1/c : c^(-σ)
u_prime_inv(c::Float64, σ::Float64) = isone(σ) ? 1/c : c^(-1/σ)

# Production function and its derivatives
function f(k::Float64, e::Float64, α::Float64)
    return k^α * e^(1-α)
end

function f_prime_k(k::Float64, e::Float64, α::Float64)
    return α * k^(α-1) * e^(1-α)
end

function f_prime_e(k::Float64, e::Float64, α::Float64)
    return (1-α) * k^α * e^(-α)
end

# Define the function to compute next capital, consumption, and efficiency
function next_val(k::Float64, e::Float64, c::Float64, params::NamedTuple)
    α = params.α
    A = params.A
    δ = params.δ
    β = params.β
    σ = params.σ

    # Calculate next capital (state transition law)
    k_next = f(k, e, α) + (1 - δ) * k - c

    # Function to solve the system using the Euler equation
    function equations!(E, x)
        c_next = x[1]
        e_next = x[2]
        
        E[1] = u_prime(c, σ) - β * u_prime(c_next, σ) * (f_prime_k(k, e_next, α) + (1 - δ))
        E[2] = β * u_prime(c_next, σ) / u_prime(c, σ) * f_prime_e(k_next, e_next, α) / f_prime_e(k, e, α) - 1
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

    for t in 1:T
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

# Multivariate Newton-Raphson method
function norm(x::Vector{Float64})
    return sqrt(sum(xi^2 for xi in x))
end

function newton_raphson(k0::Float64, c0::Float64, e0::Float64, params::NamedTuple, T::Int; tol = 1e-6, max_iter = 1e3)
    β, σ, δ, α, A, R = params.β, params.σ, params.δ, params.α, params.A, params.R

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
        e_cons = sum(e_vec[i] / A^(i-1) for i in 1:T)
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
    for i in 1:max_iter
        res = f(x)
        if any(isinf, res)
            c0 *= 0.9
            e0 *= 0.9
            x = [c0, e0]
            continue
        end
        J = jacobian(x)
        Δx = -J \ res
        x += Δx
        if norm(Δx) < tol
            return x
        end
        if i % 25 == 0
            println("Iteration $i: c0 = $c0, e0 = $e0, res = $res")
        end
    end
    println("Newton-Raphson did not converge. Returning the last guess: c0 = $c0, e0 = $e0")
    return x
end

guess = newton_raphson(1.1, 0.1, 0.1, params, 100)

# Get the original paths
c_path, k_path, e_path = shooting(1.1, guess[1], guess[2], params, 100)

# Adjust the paths by multiplying consumption and capital by A^i and dividing efficiency by A^i
function adjust_paths(c_path, k_path, e_path, A)
    T = length(c_path)
    c_adj = [c_path[i] * A^(i-1) for i in 1:T]
    k_adj = [k_path[i] * A^(i-1) for i in 1:T]
    e_adj = [e_path[i] / A^(i-1) for i in 1:T]
    return c_adj, k_adj, e_adj
end

# Adjust the paths
c_adj, k_adj, e_adj = adjust_paths(c_path, k_path, e_path, params.A)

# Function to plot the paths
function plot_paths(c_path, k_path, e_path, c_adj, k_adj, e_adj)
    plot(layout = (3, 2))
    plot!(c_path, label="Consumption", lw=2, color="blue", xlabel="Time", ylabel="Consumption", title="Intensive Consumption Path", subplot=1)
    plot!(k_path, label="Capital", lw=2, color="red", xlabel="Time", ylabel="Capital", title="Intensive Capital Path", subplot=3)
    plot!(e_path, label="Energy", lw=2, color="green", xlabel="Time", ylabel="Energy", title="Intensive Energy Path", subplot=5)
    plot!(c_adj, label="Adjusted Consumption", lw=2, color="blue", xlabel="Time", ylabel="Consumption", title="Adjusted Consumption Path", subplot=2)
    plot!(k_adj, label="Adjusted Capital", lw=2, color="red", xlabel="Time", ylabel="Capital", title="Adjusted Capital Path", subplot=4)
    plot!(e_adj, label="Adjusted Energy", lw=2, color="green", xlabel="Time", ylabel="Energy", title="Adjusted Energy Path", subplot=6)
end

# Plot the paths
plot_paths(c_path, k_path, e_path, c_adj, k_adj, e_adj)
