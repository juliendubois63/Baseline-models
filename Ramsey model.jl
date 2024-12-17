using Plots


# Define parameters as a named tuple for easy access
params = (; σ = 2.0,    # Coefficient of relative risk aversion
               β = 0.95,   # Discount factor
               δ = 0.02,   # Depreciation rate
               α = 0.33)   # Capital share

# Utility function and its derivative
u(c::Float64, σ::Float64) = isone(σ) ? log(c) : (c^(1-σ)-1)/(1-σ)
u_prime(c::Float64, σ::Float64) = isone(σ) ? 1/c : c^(-σ)
u_prime_inv(c::Float64, σ::Float64) = isone(σ) ? exp(c) : c^(-1/σ)

# Production function and its derivative
function f(k::Float64, α::Float64)
    return k^α
end

function f_prime(k::Float64, α::Float64)
    return α * k^(α-1)
end

struct RamseyModel
    params::NamedTuple
    u::Function
    u_prime::Function
    u_prime_inv::Function
    f::Function
    f_prime::Function
end

model = RamseyModel(
    params,
    u,
    u_prime,
    u_prime_inv,
    f,
    f_prime
)

# Function for state transition (next capital and consumption)
function next_k_c(k::Float64, c::Float64, model::RamseyModel)
    params = model.params
    δ = params.δ
    β = params.β
    σ = params.σ
    α = params.α
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

# Shooting method to compute consumption and capital paths for T periods
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

c_vec, k_vec = shooting(0.2, 0.3, 10, model)
test_c = plot(c_vec, label="Consumption", lw=2, color="blue", xlabel="Time", ylabel="Consumption", title="Consumption Path", show=false)
test_k = plot(k_vec, label="Capital", lw=2, color="red", xlabel="Time", ylabel="Capital", title="Capital Path", show=false)
plot(test_c, test_k, layout=(1, 2), size=(800, 400))

# Bisection method to find the optimal consumption path
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

plot(shooting(bisection(0.3, 0.3, 10, model), 0.3, 10, model)[2]) 


function compute_paths(model::RamseyModel, k0::Float64, T_arr::Vector{Int}, c0::Float64)
    c_paths = []
    k_paths = []
    lagrange_paths = []
    
    for T in T_arr
        c0_opt = bisection(c0, k0, T, model)
        result = shooting(c0_opt, k0, T, model)
        if result === nothing
            println("Failed to compute paths for T = $T")
            continue
        end
        c_vec, k_vec = result
        push!(c_paths, c_vec)
        push!(k_paths, k_vec)

        μ_vec = model.u_prime.(c_vec, model.params.σ)
        push!(lagrange_paths, μ_vec)
    end

    return c_paths, k_paths, lagrange_paths
end

# Compute the optimal consumption, capital, and Lagrange multiplier paths
c_paths, k_paths, lagrange_paths = compute_paths(model, 0.3, [100], 0.2)

# Plot the computed paths
p1 = plot(c_paths[1], label="Consumption", lw=2, color="blue", xlabel="Time", ylabel="Consumption", title="Consumption Path", show=false)
p2 = plot(k_paths[1], label="Capital", lw=2, color="red", xlabel="Time", ylabel="Capital", title="Capital Path", show=false)
p3 = plot(lagrange_paths[1], label="Lagrange Multiplier", lw=2, color="green", xlabel="Time", ylabel="Lagrange Multiplier", title="Lagrange Multiplier Path",show=false)

plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))