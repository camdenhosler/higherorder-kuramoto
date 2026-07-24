using LinearAlgebra

include("dynamics.jl")
include("utils.jl")

@inline function circular_diff(theta1::AbstractVector, theta2::AbstractVector)
    return mod.(theta1 .- theta2 .+ pi, 2pi) .- pi
end

@inline function circular_diff(theta1::AbstractVector, theta2::Real)
    return mod.(theta1 .- theta2 .+ pi, 2pi) .- pi
end


function orthogonal_proj_measures(vec1::AbstractVector, vec2::AbstractVector, target_vec::AbstractVector)
    norm1_sq = dot(vec1, vec1)
    norm2_sq = dot(vec2, vec2)

    if norm1_sq < 1e-14 || norm2_sq < 1e-14
        return (
            OutOfSpanResidual = norm(target_vec), 
            SpanResidual = 0.0, 
            TotalDist = norm((vec1 .+ vec2) .- target_vec),
            IsDependent = true)
    end

    cos_sq = (dot(vec1, vec2)^2) / (norm1_sq * norm2_sq)
    is_dependent = (1.0 - cos_sq) < 1e-12

    A = hcat(vec1, vec2)

    if is_dependent
        proj_vec = vec1 * (vec1 \ target_vec)
    else 
        proj_vec = A * (A \ target_vec)
    end

    eps1 = norm(target_vec - proj_vec)
    eps2 = norm((vec1 + vec2) - proj_vec)
    #how far the target vec is from the additive effects
    eps3 = norm((vec1 + vec2) - target_vec)

    return (
        OutOfSpanResidual = eps1, 
        SpanResidual = eps2, 
        TotalDist = eps3, 
        IsDependent = is_dependent)
end


function perturbation(node1::Int, node2::Int, theta_init::AbstractVector, params)
    pert_str = params.pert_str
    dynamic_params = Base.structdiff(params, (; pert_str=params.pert_str))

    initialEvolve = evolve_to_fixed_point(theta_init, dynamic_params)
    initial_state = initialEvolve.State

    single1_init = copy(initial_state)
    single2_init = copy(initial_state)

    single1_init[node1] += pert_str
    single2_init[node2] += pert_str

    double_init = copy(single1_init)
    double_init[node2] += pert_str

    single1Evolve = evolve_to_fixed_point(single1_init, dynamic_params)
    single2Evolve = evolve_to_fixed_point(single2_init, dynamic_params)
    doubleEvolve = evolve_to_fixed_point(double_init, dynamic_params)
    single1_state = single1Evolve.State
    single2_state = single2Evolve.State
    double_state = doubleEvolve.State

    StateVectors = (
    S1 = circular_diff(single1_state, initial_state),
    S2 = circular_diff(single2_state, initial_state),
    D = circular_diff(double_state, initial_state)
    )

    RelativeStateVectors = (
    S1 = circular_diff(StateVectors.S1, StateVectors.S1[1]),
    S2 = circular_diff(StateVectors.S2, StateVectors.S2[1]),
    D = circular_diff(StateVectors.D, StateVectors.D[1])
    )

    all_are_fixed = initialEvolve.IsFixed && single1Evolve.IsFixed && single2Evolve.IsFixed && doubleEvolve.IsFixed
    all_are_stable = initialEvolve.IsStable && single1Evolve.IsStable && single2Evolve.IsStable && doubleEvolve.IsStable

    return (
        StateVectors = StateVectors, 
        RelativeStateVectors = RelativeStateVectors, 
        AllAreFixed = all_are_fixed,
        AllAreStable = all_are_stable
    ) 
end


function calc_h_epistasis(theta_batch, node1_batch, node2_batch, params_batch)
    N = Int(params_batch.N)
    n_trials = length(node1_batch)
    
    offsets = params_batch.offsets

    nonrel_out_of_span_residual = zeros(Float64, n_trials)
    nonrel_span_residual = zeros(Float64, n_trials)
    nonrel_total_dist = zeros(Float64, n_trials)
    nonrel_is_dependent = zeros(Float64, n_trials)
    rel_out_of_span_residual = zeros(Float64, n_trials)
    rel_span_residual = zeros(Float64, n_trials)
    rel_total_dist = zeros(Float64, n_trials)
    rel_is_dependent = zeros(Float64, n_trials)
    valid_flags = zeros(Bool, n_trials)

    for t in 1:n_trials
        #since Julia begins indexing at 1 and slices inclusively on both sides
        start_idx = Int(offsets[t]) + 1
        end_idx   = Int(offsets[t+1])

        params_t = (
            omega = params_batch.omega,
            K = Float64(params_batch.K),
            N = N,
            idx_i = params_batch.idx_i[start_idx:end_idx],
            idx_j = params_batch.idx_j[start_idx:end_idx],
            idx_k = params_batch.idx_k[start_idx:end_idx],
            vals  = params_batch.vals[start_idx:end_idx],
            pert_str = Float64(params_batch.pert_str)
        )

        theta_init_t = Vector{Float64}(theta_batch[t, :])
        n1 = Int(node1_batch[t])
        n2 = Int(node2_batch[t])

        pertData = perturbation(n1, n2, theta_init_t, params_t)
        (; StateVectors, RelativeStateVectors, AllAreFixed, AllAreStable) = pertData

        if !AllAreFixed || !AllAreStable
            nonrel_out_of_span_residual[t] = NaN
            nonrel_span_residual[t] = NaN
            nonrel_total_dist[t] = NaN
            nonrel_is_dependent[t] = NaN
            rel_out_of_span_residual[t] = NaN 
            rel_span_residual[t] = NaN
            rel_total_dist[t] = NaN
            rel_is_dependent[t] = NaN
            valid_flags[t] = false
            continue
        end

        stateEpis = orthogonal_proj_measures(StateVectors.S1, StateVectors.S2, StateVectors.D)
        relStateEpis = orthogonal_proj_measures(RelativeStateVectors.S1, RelativeStateVectors.S2, RelativeStateVectors.D)

        nonrel_out_of_span_residual[t] = stateEpis.OutOfSpanResidual
        nonrel_span_residual[t] = stateEpis.SpanResidual
        nonrel_total_dist[t] = stateEpis.TotalDist
        nonrel_is_dependent[t] = stateEpis.IsDependent
        rel_out_of_span_residual[t] = relStateEpis.OutOfSpanResidual
        rel_span_residual[t] = relStateEpis.SpanResidual
        rel_total_dist[t] = relStateEpis.TotalDist
        rel_is_dependent[t] = relStateEpis.IsDependent
        valid_flags[t] = true
    end

    return (
        nonrelOutOfSpanResidual = nonrel_out_of_span_residual, 
        nonrelSpanResidual = nonrel_span_residual, 
        nonrelTotalDist = nonrel_total_dist,
        nonrelIsDependent = nonrel_is_dependent, 
        relOutOfSpanResidual = rel_out_of_span_residual, 
        relSpanResidual = rel_span_residual, 
        relTotalDist = rel_total_dist,
        relIsDependent = rel_is_dependent, 
        validFlags = valid_flags)
end
