from typing import Tuple, Dict, Any
from functools import partial
import numpy as np

def perturbation(node1: int, node2: int, perturb_strength: float, t_start: float, t_end: float, 
                 theta_init: np.ndarray, params: Dict[str, Any], func) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    cut_find_fpas = partial(func,t_start=t_start,t_end=t_end,params=params)

    _, _, ifpa, fixed1, stab1 = cut_find_fpas(theta_init=theta_init)

    fs_perturb = ifpa.copy()
    ss_perturb = ifpa.copy()

    fs_perturb[node1] = fs_perturb[node1] + perturb_strength
    ss_perturb[node2] = ss_perturb[node2] + perturb_strength

    d_perturb = fs_perturb.copy()
    d_perturb[node2] = fs_perturb[node2] + perturb_strength

    _, _, fsfpa, fixed2, stab2 = cut_find_fpas(theta_init=fs_perturb)
    _, _, ssfpa, fixed3, stab3 = cut_find_fpas(theta_init=ss_perturb)
    _, _, dfpa, fixed4, stab4 = cut_find_fpas(theta_init=d_perturb)

    def angular_diff(theta1, theta2):
        return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi

    dists = {
        'fs': fsfpa - ifpa,
        'ss': ssfpa - ifpa,
        'd': dfpa - ifpa,
    }

    diff_dists = {
        'fs': angular_diff(dists['fs'], dists['fs'][0]),
        'ss': angular_diff(dists['ss'], dists['ss'][0]),
        'd': angular_diff(dists['d'], dists['d'][0]),
    }

    total_fixed = fixed1 & fixed2 & fixed3 & fixed4
    total_stability = stab1 & stab2 & stab3 & stab4

    total_fs = total_fixed & total_stability

    return dists, diff_dists, total_fs

def projection_distance(fpa1: np.ndarray, fpa2: np.ndarray, target_fpa: np.ndarray):
    A = np.column_stack((fpa1,fpa2))
    #calculate moore penrose pseudo inverses
    A_pinv = np.linalg.pinv(A)
    Proj_target = A @ A_pinv @ target_fpa

    orth_distance = np.linalg.norm(target_fpa - Proj_target)
    span_distance = np.linalg.norm((fpa1 + fpa2) - Proj_target)

    rank = np.linalg.matrix_rank(np.column_stack((fpa1, fpa2)))

    if rank < 2:
        degenerate = True
    else:
        degenerate = False

    return orth_distance, span_distance, degenerate