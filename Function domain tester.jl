function f_k(k::Float64, e::Float64, α::Float64, ϵ::Float64, γ::Float64)
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

params = (; σ = 1.0,    # Coefficient of relative risk aversion
               β = 0.985,   # Discount factor
               δ = 0.02,   # Depreciation rate
               α = 0.2685,    # Capital share
               ϵ = 0.02,   # Elasticity of substitution
               Ae = 1.03,    # Energy saving TP
               A = 1.02,    # Capital TP
               R = 10)
using ForwardDiff

# Function to verify the domain of `f`
function verify_domain(f, k_range, e_range, α, ϵ; num_points=1000)
    k_values = range(k_range[1], k_range[2], length=num_points)
    e_values = range(e_range[1], e_range[2], length=num_points)
    defined_points = []  # Store all valid (k, e) points

    for k in k_values
        for e in e_values
            try
                # Try evaluating the function
                val = f(k, e, α, ϵ, 0.05)
                if isfinite(val)  # Ensure the value is finite
                    push!(defined_points, (k, e))
                end
            catch
                # If any error occurs, ignore the point
                continue
            end
        end
    end

    # Extract bounds from defined points
    if !isempty(defined_points)
        k_bounds = (minimum(first.(defined_points)), maximum(first.(defined_points)))
        e_bounds = (minimum(last.(defined_points)), maximum(last.(defined_points)))
    else
        k_bounds = nothing
        e_bounds = nothing
    end

    return k_bounds, e_bounds, defined_points
end

# Range of k and e to check
k_range = (-5.0, 5.0)
e_range = (-5.0, 5.0)

# Verify the domain
k_bounds, e_bounds, defined_points = verify_domain(f_k, k_range, e_range, params.α, params.ϵ)

# Print the results
if k_bounds !== nothing && e_bounds !== nothing
    println("Function f is defined within the following bounds:")
    println("k bounds: $k_bounds")
    println("e bounds: $e_bounds")
else
    println("Function f is not defined within the given ranges.")
end
