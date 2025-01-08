using Plots

params = (; σ = 2.0,    # Coefficient of relative risk aversion
               β = 0.95,   # Discount factor
               δ = 0.02,   # Depreciation rate
               α = 0.33,    # Capital share
               A = 1.05) # Total factor productivity

# Utility function and its derivative
u(c::Float64, σ::Float64) = isone(σ) ? log(c) : (c^(1-σ)-1)/(1-σ)
u_prime(c::Float64, σ::Float64) = isone(σ) ? 1/c : c^(-σ)
u_prime_inv(c::Float64, σ::Float64) = isone(σ) ? 1/c : c^(-1/σ)

# Production function and its derivative
function f(k::Float64, α::Float64)
    return k^α
end

function f_prime(k::Float64, α::Float64)
    return α * k^(α-1)
end

function f_prime_inv(k::Float64, α::Float64)
    return (k / (α))^(1/(α-1))
end

struct RamseyModel
    params::NamedTuple
    u::Function
    u_prime::Function
    u_prime_inv::Function
    f::Function
    f_prime::Function
end

model = RamseyModel(params, u, u_prime, u_prime_inv, f, f_prime)

# Function for state transition (next capital and consumption)
function next_k_c(k::Float64, c::Float64, model::RamseyModel)
    params = model.params
    δ = params.δ
    β = params.β
    σ = params.σ
    α = params.α
    A = params.A
    u_prime = model.u_prime
    u_prime_inv = model.u_prime_inv
    f = model.f
    f_prime = model.f_prime

    # Calculate next capital (state transition law)
    k_next = f(k, α) + (1 - δ) * k - c
    # Calculate next consumption from Euler equation
    c_next = u_prime_inv(u_prime(c, σ) / (β * (f_prime(k_next, α) + (1 - δ))), σ)

    return k_next, c_next
end

function shooting(c0::Float64, k0::Float64, T::Int, model::RamseyModel)
    c_vec = zeros(Float64, T + 1)
    k_vec = zeros(Float64, T + 2)
    c_vec[1] = c0
    k_vec[1] = k0

    for t in 1:T
        k_next, c_next = next_k_c(k_vec[t], c_vec[t], model)
                
        k_vec[t + 1] = k_next
        c_vec[t + 1] = c_next
    end

    # Compute terminal capital using the resource constraint
    k_terminal = f(k_vec[T + 1], model.params.α) + (1 - model.params.δ) * k_vec[T + 1] - c_vec[T + 1]
    
    # Ensure that the terminal capital is positive
    if k_terminal <= 0
        return nothing
    end
    
    k_vec[T + 2] = k_terminal

    return c_vec, k_vec
end

function bisection(c0::Float64, k0::Float64, T::Int, model::RamseyModel; tol::Float64 = 1e-6, max_iter::Int = 1000)
    c_upp = model.f(k0, model.params.α) + (1 - model.params.δ) * k0 
    c_low = 0.0
    i = 0
    
    while i < max_iter
        result = try
            shooting(c0, k0, T, model)
        catch e
            if isa(e, DomainError)
                c_upp = c0
                c0 = (c_upp + c_low) / 2
                continue
            else
                rethrow(e)
            end
        end

        if result === nothing
            c_upp = c0
            c0 = (c_upp + c_low) / 2
            continue
        end

        c_vec, k_vec = result
        error = k_vec[end]  # Ensure terminal capital is zero

        if abs(error) < tol
            return c0
        elseif error > 0
            c_low = c0
        else
            c_upp = c0
        end
        c0 = (c_upp + c_low) / 2
        i += 1
    end 
    return c0
end

function bisection_inf(c0::Float64, k0::Float64, T::Int, model::RamseyModel; tol::Float64 = 1e-6, max_iter::Int = 1000)
    c_upp = model.f(k0, model.params.α, model.params.A) + (1 - model.params.δ) * k0 
    c_low = 0.0
    i = 0
   
    while i < max_iter
        i += 1
        result = try
            shooting(c0, k0, T, model)
        catch e
            if isa(e, DomainError)
                c_upp = c0
                c0 = (c_upp + c_low) / 2
                continue
            else
                rethrow(e)
            end
        end

        if result === nothing
            c_upp = c0
            c0 = (c_upp + c_low) / 2
            continue
        end

        c_vec, k_vec = result
        error = model.u_prime(c_vec[end], model.params.σ) - model.params.β * model.u_prime(c_vec[end-1], model.params.σ) * (1 + model.f_prime(k_vec[end-1], model.params.α, model.params.A) - model.params.δ) # Ensure TVC

        if abs(error) < tol
            return c0
        elseif error > 0
            c_low = c0
        else
            c_upp = c0
        end
        c0 = (c_upp + c_low) / 2
    end 
    return c0
end

# Function to compute technology-adjusted paths
function compute_tech_adjusted_paths(c0::Float64, k0::Float64, T::Int, model::RamseyModel)
    c_opt = bisection_inf(c0, k0, T, model)
    c_vec, k_vec = shooting(c_opt, k0, T, model)

    A = model.params.A
    k_adj_vec = [k_vec[i] * A^i for i in 1:length(k_vec)]

    c_adj_vec = zeros(Float64, length(c_vec))

    for i in 1:length(c_vec)-1
        c_adj_vec[i] = model.f(k_adj_vec[i], model.params.α) + (1 - model.params.δ) * k_adj_vec[i] - k_adj_vec[i+1]
    end

    return k_adj_vec, c_adj_vec
end

function plot_all_paths(c_vec, k_vec, k_adj_vec, c_adj_vec, T::Int)
    p1 = plot(1:T+1, c_vec, label="Consumption", xlabel="Time", ylabel="Value", title="Consumption (Non-Adjusted)", color="blue")
    p2 = plot(1:T+1, k_vec[1:end-1], label="Capital", xlabel="Time", ylabel="Value", title="Capital (Non-Adjusted)", color="red")
    p3 = plot(1:T, c_adj_vec[1:T], label="Adjusted Consumption", xlabel="Time", ylabel="Value", title="Adjusted Consumption", color="blue")
    p4 = plot(1:T+1, k_adj_vec[1:end-1], label="Adjusted Capital", xlabel="Time", ylabel="Value", title="Adjusted Capital", color="red")

    plot(p1, p2, p3, p4, layout=(2, 2))
end

# Initial values
c0 = 0.1
k0 = 1.0
T = 100

# Compute paths
c_opt = bisection_inf(c0, k0, T, model)
c_vec, k_vec = shooting(c_opt, k0, T, model)
k_adj_vec, c_adj_vec = compute_tech_adjusted_paths(c0, k0, T, model)

# Plot all paths in a 2x2 layout
plot_all_paths(c_vec, k_vec, k_adj_vec, c_adj_vec, T)
# Compute paths for finite time
c_opt_finite = bisection(c0, k0, T, model)
c_vec_finite, k_vec_finite = shooting(c_opt_finite, k0, T, model)

# Plot all paths in a 2x2 layout for infinite time
plot_all_paths(c_vec, k_vec, k_adj_vec, c_adj_vec, T)

# Plot all paths in a 2x2 layout for finite time
plot_all_paths(c_vec_finite, k_vec_finite, k_vec_finite, c_vec_finite, T)
