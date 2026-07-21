from typing import Tuple, Dict, Any
import numpy as np
import numba as nb
from scipy.integrate import solve_ivp
from scipy.optimize import root
import scipy.linalg as la

@nb.njit(cache=True)
def build_sparse_A(A):
    idx_i, idx_j = np.nonzero(A)
    vals = np.empty(len(idx_i))
    for e in range(len(idx_i)):
        vals[e] = A[idx_i[e], idx_j[e]]
    return idx_i, idx_j, vals

@nb.njit(fastmath=True, cache=True)
def kuramoto_func(_: float, theta: np.ndarray, omega: np.ndarray, K: float, 
                         N: int, idx_i: np.ndarray, idx_j: np.ndarray, vals: np.ndarray) -> np.ndarray:
    theta_deriv = np.zeros(N)
    n_edges = len(vals)
    for e in range(n_edges):
        i = idx_i[e]
        j = idx_j[e]

        theta_deriv[i] += vals[e] * np.sin(theta[j] - theta[i]) + vals[e] * ( K / N ) * np.sin(2 * (theta[j] - theta[i]))
    return theta_deriv

@nb.njit(fastmath=True, cache=True)
def jacobian(theta: np.ndarray, K: float, N: int, idx_i: np.ndarray, 
             idx_j: np.ndarray, vals: np.ndarray)-> np.ndarray:
    J = np.zeros((N, N))
    n_edges = len(vals)

    for e in range(n_edges):
        i = idx_i[e]
        j = idx_j[e]

        arg = theta[j] - theta[i]

        dy1_cos = np.cos(arg)
        dy2_cos = np.cos(2 * arg)

        term1_der = vals[e] * dy1_cos
        term2_der = vals[e] * ( 2 * K / N ) * dy2_cos

        d_di = -term1_der - term2_der
        d_dj = term1_der + term2_der

        J[i, i] += d_di
        J[i, j] += d_dj

    return J

def find_fpas(t_start: float,t_end: float, theta_init: np.ndarray, 
              params: Dict[str, Any], func=kuramoto_func) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Integrates the system to find approximate root which is refined by scipy
    root and then checks for stability through the eigenvalues of jacobian.
    Note that the jacobian is calculated only for the regularized kuramoto
    no other dynamics.
    """
    idx_i, idx_j, vals = params['sparse_A']

    solution = solve_ivp(
        fun=func,
        t_span=(t_start, t_end),
        y0=theta_init,
        args=(params['omega'], params['K'], params['N'],
              idx_i, idx_j, vals),
        method='LSODA',
        rtol=1e-5,
        atol=1e-7,
    )

    print(f"Int finished at {solution.t[-1]}")

    candidate_phases = solution.y[:, -1]

    def root_target(theta):
        return func(0.0,theta,params['omega'],params['K'],params['N'], 
                    idx_i, idx_j, vals)

    res = root(root_target, candidate_phases, method='hybr')

    if not res.success:
        print("NOT FIXED")
        candidate_derivs = root_target(candidate_phases)
        return candidate_phases % (2 * np.pi), candidate_derivs, candidate_phases, False

    fixed_point = res.x
    fixed_derivs = root_target(fixed_point)

    J = jacobian(fixed_point, params['K'], params['N'],
                  idx_i, idx_j, vals)
    eigenvalues = la.eigvals(J)

    real_eigvalues = np.real(eigenvalues)
    max_eigval = np.max(real_eigvalues)

    stability_threshold = 1e-3

    if max_eigval > stability_threshold:
        print("UNSTABLE")
        return fixed_point % (2 * np.pi), fixed_derivs, fixed_point, False

    final_phases = fixed_point
    final_phases_mod = final_phases % (2 * np.pi)
    final_derivs = fixed_derivs

    return final_phases_mod, final_derivs, final_phases, True

