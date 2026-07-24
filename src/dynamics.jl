using LinearAlgebra: eigvals, norm
using ForwardDiff: jacobian
using DifferentialEquations
using NonlinearSolve
using DiffEqCallbacks


function h_kuramoto_func!(dtheta::AbstractVector, theta::AbstractVector, params, _)
    (; K, omega, N, idx_i, idx_j, idx_k, vals) = params

    n_edges = length(vals)

    dtheta .= omega 

    @inbounds for e in 1:n_edges
        i = idx_i[e]
        j = idx_j[e]
        k = idx_k[e]

        phase_diff = theta[k] + theta[j] - 2 * theta[i]
        dtheta[i] +=  vals[e] * sin(phase_diff) + (K / float(N)) * vals[e] * sin(phase_diff) * cos(theta[j] - theta[k])
    end

    return nothing
end

function integrate_to_candidate(theta_init::AbstractVector, params)

    prob = ODEProblem(
        h_kuramoto_func!,
        theta_init,
        (0.0, 1000.0),
        params
    )

    cb = TerminateSteadyState(1e-3, 1e-3)
    sol = solve(
        prob,
        AutoTsit5(TRBDF2()),
        callback=cb,
        reltol=1e-5,
        abstol=1e-7
    )

    println("Number of steps: ", length(sol.t))
    return copy(sol.u[end])
end

function refine_fixed_point!(candidate::AbstractVector, params)
    #rewrite for non zero omegas
    f_root!(du, u, p) = h_kuramoto_func!(du, u, p, 0.0)
    prob = NonlinearProblem(f_root!, candidate, params)

    sol = solve(
        prob, 
        TrustRegion(), 
        reltol=1e-5, 
        abstol=1e-7,
        )
    
    candidate .= sol.u

    dtheta_dt = similar(candidate)
    f_root!(dtheta_dt, candidate, params)

    linf_err = norm(dtheta_dt, Inf)

    is_fixed = linf_err < 1e-5

    return (Candidate = candidate, IsFixed = is_fixed)
end

function stability_check(fixed_point::AbstractVector, params)
    N = params.N
    out_dtheta = zeros(N)
    out_J = zeros(N, N) 

    f_closure! = (dtheta_vec, theta_vec) -> h_kuramoto_func!(dtheta_vec, theta_vec, params, 0.0)

    cfg = ForwardDiff.JacobianConfig(f_closure!, out_dtheta, fixed_point)
    ForwardDiff.jacobian!(out_J, f_closure!, out_dtheta, fixed_point, cfg)

    eigs = eigvals(out_J)
    real_eigs = real.(eigs)
    sorted_reals = sort(real_eigs, rev=true)
    zero_idx = argmin(abs.(sorted_reals))
    physical_eigs = deleteat!(copy(sorted_reals), zero_idx)

    is_stable = maximum(physical_eigs) < -1e-5 && abs(sorted_reals[zero_idx]) < 1e-3

    return (Jacobian = out_J, Eigenvalues = eigs, IsStable = is_stable)
end

function evolve_to_fixed_point(theta_init::AbstractVector, params)
    candidate = integrate_to_candidate(theta_init, params)
    rootData = refine_fixed_point!(candidate, params)
    is_fixed = rootData.IsFixed
    
    if !is_fixed
        dtheta = similar(candidate)
        h_kuramoto_func!(dtheta, candidate, params, 0.0)
        return (State = candidate, Deriv = dtheta, IsFixed = is_fixed, IsStable = false, Jacobian = nothing)
    end

    fixed_point = rootData.Candidate
    stabData = stability_check(fixed_point, params)
    is_stable = stabData.IsStable

    dtheta = similar(fixed_point)
    h_kuramoto_func!(dtheta, fixed_point, params, 0.0)
    return (State = fixed_point, Deriv = dtheta, IsFixed = is_fixed, IsStable = is_stable, Jacobian = stabData.Jacobian)
end