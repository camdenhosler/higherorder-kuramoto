from typing import Tuple, Dict, Any
import numpy as np
from scipy.integrate import solve_ivp

def kuramoto_func(_: float, theta: np.ndarray, omega: np.ndarray, K: float, N: int, A: np.ndarray) -> np.ndarray:
    diff = theta[None,:] - theta[:,None]
    #maybe can replace the sum with @ (normal matrix multiplication not the pairwise *)
    #theta_deriv =  np.sum(A * np.sin(diff),axis=1) + K/N * np.sum(np.sin(2*diff),axis=1)
    theta_deriv =  np.sum(A * np.sin(diff),axis=1) + K/N * np.sum(A * np.sin(2*diff),axis=1)
    return theta_deriv

def find_fpas(t_start: float,t_end: float, theta_init: np.ndarray, t_eval: np.ndarray,  params: Dict[str, Any], func=kuramoto_func) -> Tuple[np.ndarray, np.ndarray]:

    solution = solve_ivp(
        fun=func,
        t_span=(t_start, t_end),
        y0=theta_init,
        t_eval=t_eval,
        args=(params['omega'], params['K'], params['N'], params['A']),
        method='RK45',
        rtol=1e-6,
        atol=1e-7
    )

    final_phases = solution.y[:, -1]
    final_phases_mod = final_phases % (2 * np.pi)

    final_derivs = func(t_end, final_phases, params['omega'], params['K'], params['N'], params['A'])

    return final_phases_mod, final_derivs
