from typing import Tuple, Dict, Any
import numpy as np
import numba as nb
from scipy.integrate import solve_ivp
from scipy.optimize import root
import scipy.linalg as la

@nb.njit(cache=True)
def build_sparse_A(A):
    idx_i, idx_j, idx_k = np.nonzero(A)
    vals = np.empty(len(idx_i))
    for e in range(len(idx_i)):
        vals[e] = A[idx_i[e], idx_j[e], idx_k[e]]
    return idx_i, idx_j, idx_k, vals

@nb.njit(fastmath=True, cache=True)
def higher_kuramoto_func(_: float, theta: np.ndarray, omega: np.ndarray, K: float, 
                         N: int, idx_i: np.ndarray, idx_j: np.ndarray, idx_k: np.ndarray,
                         vals: np.ndarray) -> np.ndarray:
    theta_deriv = omega.copy()
    n_edges = len(vals)
    for e in range(n_edges):
        i = idx_i[e]
        j = idx_j[e]
        k = idx_k[e]
        #theta_deriv[i] += vals[e] * np.sin(theta[k] + theta[j] - 2 * theta[i]) + K/N * np.sin(2 * (theta[k] + theta[j] - 2 * theta[i]))
        theta_deriv[i] += vals[e] * np.sin(theta[k] + theta[j] - 2 * theta[i]) + vals[e] * 2 * (K / N) * np.sin(theta[k] + theta[j] - 2 * theta[i]) * np.cos(theta[j] - theta[k])
        #theta_deriv[i] += (K / N) * vals[e] * np.sin(theta[k] + theta[j]  - 2 * theta[i])
        
    return theta_deriv

def zero_deriv(_: float, theta: np.ndarray, omega: np.ndarray, K: float, N: int, idx_i: np.ndarray, idx_j: np.ndarray, idx_k: np.ndarray, vals: np.ndarray):
    theta_deriv = higher_kuramoto_func(_, theta, omega, K, N, idx_i, idx_j, idx_k, vals)
    inf_norm = np.max(np.abs(theta_deriv))
    return inf_norm - 0.001

@nb.njit(fastmath=True, cache=True)
def jacobian(theta: np.ndarray, K: float, N: int, idx_i: np.ndarray, 
             idx_j: np.ndarray, idx_k: np.ndarray, vals: np.ndarray)-> np.ndarray:
    #for high N this will become very slow
    J = np.zeros((N, N))
    n_edges = len(vals)

    for e in range(n_edges):
        i = idx_i[e]
        j = idx_j[e]
        k = idx_k[e]

        arg_sin = theta[k] + theta[j] - 2 * theta[i]
        arg_cos = theta[j] - theta[k]

        h_cos = np.cos(arg_sin)
        h_sin = np.sin(arg_sin)
        dy_cos = np.cos(arg_cos)
        dy_sin = np.sin(arg_cos)

        term1_der = vals[e] * h_cos
        term2_der_p1 = ( 2 * K / N ) * h_cos * dy_cos
        term2_der_p2 = ( 2 * K / N ) * h_sin * dy_sin

        d_di = -2 * ( term1_der + term2_der_p1)
        d_dj = term1_der + term2_der_p1 - term2_der_p2
        d_dk = term1_der + term2_der_p1 + term2_der_p2

        J[i, i] += d_di
        J[i, j] += d_dj
        J[i, k] += d_dk

    return J

def find_fpas(t_start: float,t_end: float, theta_init: np.ndarray, 
              params: Dict[str, Any], func=higher_kuramoto_func) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """
    Integrates the system to find approximate root which is refined by scipy
    root and then checks for stability through the eigenvalues of jacobian.
    Note that the jacobian is calculated only for the regularized kuramoto
    no other dynamics.
    """
    zero_deriv.terminal = True    
    zero_deriv.direction = 0
    idx_i, idx_j, idx_k, vals = params['sparse_A']

    solver_args = (params['omega'], params['K'], params['N'], idx_i, idx_j, idx_k, vals)

    solution = solve_ivp(
        fun=func,
        t_span=(t_start, t_end),
        y0=theta_init,
        args=solver_args,
        method='LSODA',
        rtol=1e-5,
        atol=1e-7,
        events=zero_deriv,
    )

    event_failed = len(solution.t_events[0]) == 0
    candidate_phases = solution.y[:, -1]

    def root_target(theta):
        return func(0.0,theta,params['omega'],params['K'],params['N'], 
                    idx_i, idx_j, idx_k, vals)

    res = root(root_target, candidate_phases, method='hybr')

    if not res.success or event_failed:
        print("NOT FIXED")
        candidate_derivs = root_target(candidate_phases)
        return candidate_phases % (2 * np.pi), candidate_derivs, candidate_phases, False, False

    fixed_point = res.x
    fixed_derivs = root_target(fixed_point)

    J = jacobian(fixed_point, params['K'], params['N'],
                  idx_i, idx_j, idx_k, vals)
    eigenvalues = la.eigvals(J)

    real_eigvalues = np.real(eigenvalues)
    max_eigval = np.max(real_eigvalues)

    stability_threshold = 1e-3

    if max_eigval > stability_threshold:
        print("UNSTABLE")
        return fixed_point % (2 * np.pi), fixed_derivs, fixed_point, True, False

    final_phases = fixed_point
    final_phases_mod = final_phases % (2 * np.pi)
    final_derivs = fixed_derivs

    return final_phases_mod, final_derivs, final_phases, True, True
