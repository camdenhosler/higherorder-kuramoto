using LinearAlgebra: norm
using Statistics
using Base.Threads

include("dynamics.jl")

function hdyadic_kuramoto_func!(dtheta::AbstractVector, theta::AbstractVector, params, _)
    (; omega, K, N, idx_i, idx_j, idx_k, vals) = params

    n_edges = length(vals)

    dtheta .= omega 

    @inbounds for e in 1:n_edges
        i = idx_i[e]
        j = idx_j[e]
        k = idx_k[e]

        phase_diff = theta[k] + theta[j] - 2 * theta[i]
        dtheta[i] +=  (2 * K / float(N)) * vals[e] * sin(phase_diff) * cos(theta[j] - theta[k])
    end

    return nothing
end

function calc_dyadic_domination(theta_batch, params_batch, n_trials_norm)
    N = Int(params_batch.N)
    n_trials_K = length(params_batch.K)

    offsets = params_batch.offsets
    averaged_diff_norm_vec = zeros(Float64, n_trials_K)

     @threads for i in 1:n_trials_K
        diff_norm_vec = zeros(Float64, n_trials_norm)

        @threads for j in 1:n_trials_norm
            #since Julia begins indexing at 1 and slices inclusively on both sides
            start_idx = Int(offsets[j]) + 1
            end_idx   = Int(offsets[j+1])

            params_ij = (
                omega = params_batch.omega,
                K = Float64(params_batch.K[i]),
                N = N,
                idx_i = @view(params_batch.idx_i[start_idx:end_idx]),
                idx_j = @view(params_batch.idx_j[start_idx:end_idx]),
                vals  = @view(params_batch.vals[start_idx:end_idx]),
            )

            if haskey(params_batch, :idx_k)
                params_ij = merge(params_ij, (
                    idx_k = @view(params_batch.idx_k[start_idx:end_idx]),
                ))
            end

            theta_init_j = Vector{Float64}(theta_batch[j, :])
            higherData = evolve_to_fixed_point(theta_init_j, params_ij, h_kuramoto_func!)
            regularizeTermsData = evolve_to_fixed_point(theta_init_j, params_ij, hdyadic_kuramoto_func!)
            h_state = higherData.State
            hdyadic_state = regularizeTermsData.State

            both_fixed = higherData.IsFixed && regularizeTermsData.IsFixed
            both_stable = higherData.IsStable && regularizeTermsData.IsStable

            if !both_stable || !both_fixed
                diff_norm_vec[j] = NaN
                continue
            end

            diff_norm = norm(h_state - hdyadic_state, 2)
            diff_norm_vec[j] = diff_norm
        end
        valid_vals = filter(!isnan, diff_norm_vec)
        averaged_diff_norm = isempty(valid_vals) ? NaN : mean(valid_vals)
        averaged_diff_norm_vec[i] = averaged_diff_norm
    end

    return averaged_diff_norm_vec
end