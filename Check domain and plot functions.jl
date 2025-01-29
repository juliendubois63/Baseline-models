using Plots

params = (; σ = 1.0,    # Coefficient of relative risk aversion
               β = 0.985,   # Discount factor
               δ = 0.02,   # Depreciation rate
               α = 0.2685,    # Capital share
               ϵ = 0.02,   # Elasticity of substitution
               Ae = 1.03,    # Energy saving TP
               A = 1.02,    # Capital TP
               R = 10)

# Create a new params with the updated β value
params = (; params..., β = params.β*(params.A^((1 - params.σ)/(1-params.α))), g = params.Ae/(params.A)^(1/(1-params.α)))

γ = 1e-3

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

function f_extension(k::Float64, e::Float64, α::Float64, ϵ::Float64)
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
function f_prime_e_extension(k::Float64, e::Float64, α::Float64, ϵ::Float64)
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
function f_prime_k_extension(k::Float64, e::Float64, α::Float64, ϵ::Float64)
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

# Generic function to plot any function with one variable fixed
function plot_function_with_fixed_var(f::Function, fixed_var::Symbol, fixed_val::Float64, α::Float64, ϵ::Float64, γ::Float64)
    if fixed_var == :k
        var_values = -5:0.1:10
        f_values = Float64[]
        for var in var_values
            try
                push!(f_values, f(var, fixed_val, α, ϵ))
            catch e
                if isa(e, DomainError)
                    push!(f_values, NaN)
                else
                    rethrow(e)
                end
            end
        end
        plot(var_values, f_values, label="k = $fixed_val", xlabel="e", ylabel="f(k, e)", title="$(nameof(f)) with k fixed")
    elseif fixed_var == :e
        var_values = -5:0.1:5
        f_values = Float64[]
        for var in var_values
            try
                push!(f_values, f(fixed_val, var, α, ϵ))
            catch e
                if isa(e, DomainError)
                    push!(f_values, NaN)
                else
                    rethrow(e)
                end
            end
        end
        plot(var_values, f_values, label="e = $fixed_val", xlabel="k", ylabel="f(k, e)", title="$(nameof(f)) with e fixed")
    else
        error("Invalid fixed variable. Use :k or :e.")
    end
end

function generate_all_plots(params, γ)
    p1 = plot_function_with_fixed_var(f, :k, 1.0, params.α, params.ϵ, γ)
    p2 = plot_function_with_fixed_var(f_extension, :k, 1.0, params.α, params.ϵ, γ)
    p3 = plot_function_with_fixed_var(f, :e, 1.0, params.α, params.ϵ, γ)
    p4 = plot_function_with_fixed_var(f_extension, :e, 1.0, params.α, params.ϵ, γ)
    p5 = plot_function_with_fixed_var(f_prime_e, :k, 1.0, params.α, params.ϵ, γ)
    p6 = plot_function_with_fixed_var(f_prime_e_extension, :k, 1.0, params.α, params.ϵ, γ)
    p7 = plot_function_with_fixed_var(f_prime_e, :e, 1.0, params.α, params.ϵ, γ)
    p8 = plot_function_with_fixed_var(f_prime_e_extension, :e, 1.0, params.α, params.ϵ, γ)
    p9 = plot_function_with_fixed_var(f_prime_k, :k, 1.0, params.α, params.ϵ, γ)
    p10 = plot_function_with_fixed_var(f_prime_k_extension, :k, 1.0, params.α, params.ϵ, γ)
    p11 = plot_function_with_fixed_var(f_prime_k, :e, 1.0, params.α, params.ϵ, γ)
    p12 = plot_function_with_fixed_var(f_prime_k_extension, :e, 1.0, params.α, params.ϵ, γ)

    plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, layout=(6, 2), size=(1000, 700))
end

generate_all_plots(params, γ)

function plot_function_with_multiple_fixed_vals(f::Function, f_extension::Function, fixed_var::Symbol, fixed_vals::Vector{Float64}, α::Float64, ϵ::Float64, γ::Float64)
    n = length(fixed_vals)
    plots = []
    for fixed_val in fixed_vals
        if fixed_var == :k
            var_values = -0.5:0.1:20
            f_values = Float64[]
            f_extension_values = Float64[]
            for var in var_values
                try
                    push!(f_values, f(var, fixed_val, α, ϵ))
                catch e
                    if isa(e, DomainError)
                        push!(f_values, NaN)
                    else
                        rethrow(e)
                    end
                end
                try
                    push!(f_extension_values, f_extension(var, fixed_val, α, ϵ))
                catch e
                    if isa(e, DomainError)
                        push!(f_extension_values, NaN)
                    else
                        rethrow(e)
                    end
                end
            end
            push!(plots, plot(var_values, f_values, label="k = $fixed_val", xlabel="e", ylabel="f(k, e)", title="f with k fixed"))
            push!(plots, plot(var_values, f_extension_values, label="k = $fixed_val", xlabel="e", ylabel="f_extension(k, e)", title="f_ext with k fixed"))
        elseif fixed_var == :e
            var_values = 0:0.1:20
            f_values = Float64[]
            f_extension_values = Float64[]
            for var in var_values
                try
                    push!(f_values, f(fixed_val, var, α, ϵ))
                catch e
                    if isa(e, DomainError)
                        push!(f_values, NaN)
                    else
                        rethrow(e)
                    end
                end
                try
                    push!(f_extension_values, f_extension(fixed_val, var, α, ϵ))
                catch e
                    if isa(e, DomainError)
                        push!(f_extension_values, NaN)
                    else
                        rethrow(e)
                    end
                end
            end
            push!(plots, plot(var_values, f_values, label="e = $fixed_val", xlabel="k", ylabel="f(k, e)", title="f with e fixed"))
            push!(plots, plot(var_values, f_extension_values, label="e = $fixed_val", xlabel="k", ylabel="f_extension(k, e)", title="f_ext with e fixed"))
        else
            error("Invalid fixed variable. Use :k or :e.")
        end
    end
    plot(plots..., layout=(n, 2), size=(800, 200 * n))
end

# Example usage:
fixed_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
p2 = plot_function_with_multiple_fixed_vals(f_prime_e, f_prime_e_extension, :k, fixed_vals, params.α, params.ϵ, γ)
