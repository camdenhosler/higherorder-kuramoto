from typing import Tuple, Dict, Any
import numpy as np
from scipy.integrate import solve_ivp
import numba as nb

@nb.njit(cache=True)
def build_sparse_A(A):
    idx_i, idx_j, idx_k = np.nonzero(A)
    vals = np.empty(len(idx_i))
    for e in range(len(idx_i)):
        vals[e] = A[idx_i[e], idx_j[e], idx_k[e]]
    return idx_i, idx_j, idx_k, vals

@nb.njit(fastmath=True, cache=True)
def kuramoto_func(_: float, theta: np.ndarray, omega: np.ndarray, K: float, 
                         N: int, idx_i: np.ndarray, idx_j: np.ndarray, idx_k: np.ndarray,
                         vals: np.ndarray) -> np.ndarray:
    theta_deriv = np.zeros(N)
    # n_edges = len(vals)
    # for e in range(n_edges):
    #     i = idx_i[e]
    #     j = idx_j[e]
    #     k = idx_k[e]

    #     theta_deriv[i] += vals[e] * np.sin(theta[j] - theta[i]) + ( K / N ) * np.sin(2 * (theta[j] - theta[i]))
    #maybe can replace the sum with @ (normal matrix multiplication not the pairwise *)
    #theta_deriv =  np.sum(A * np.sin(diff),axis=1) + K/N * np.sum(np.sin(2*diff),axis=1)
    #theta_deriv =  np.sum(A * np.sin(diff),axis=1) + K/N * np.sum(A * np.sin(2*diff),axis=1)
    return theta_deriv

def derivative_zero(t, theta: np.ndarray, omega: np.ndarray, K: float, 
                         N: int, A: np.ndarray):
    raw_dthetadt = kuramoto_func(t, theta, omega, K, N, A)
    abs_dthetadt = np.abs(raw_dthetadt)
    inf_norm = np.max(abs_dthetadt)
    
    threshold = 0.001
    return inf_norm - threshold

def find_fpas(t_start: float,t_end: float, theta_init: np.ndarray, t_eval: np.ndarray,  params: Dict[str, Any], func=kuramoto_func) -> Tuple[np.ndarray, np.ndarray]:

    derivative_zero.terminal = True
    derivative_zero.direction = 0 

    solution = solve_ivp(
        fun=func,
        t_span=(t_start, t_end),
        y0=theta_init,
        t_eval=t_eval,
        args=(params['omega'], params['K'], params['N'], params['A']),
        method='LSODA',
        rtol=1e-6,
        atol=1e-7,
        events=[derivative_zero],
    )

    print(f"Int finished at {solution.t[-1]}")

    final_phases = solution.y[:, -1]
    final_phases_mod = final_phases % (2 * np.pi)

    final_derivs = func(t_end, final_phases, params['omega'], params['K'], params['N'], params['A'])
    return final_phases_mod, final_derivs
