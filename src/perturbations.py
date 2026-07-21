from typing import Tuple, Dict, Any
from functools import partial
import numpy as np

def perturbation(node1: int, node2: int, perturb_strength: float, t_start: float, t_end: float, 
                 theta_init: np.ndarray, params: Dict[str, Any], func) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    cut_find_fpas = partial(func,t_start=t_start,t_end=t_end,params=params)

    m_ifpa, _, ifpa, stab1 = cut_find_fpas(theta_init=theta_init)

    fs_perturb = ifpa.copy()
    ss_perturb = ifpa.copy()

    fs_perturb[node1] = fs_perturb[node1] + perturb_strength
    ss_perturb[node2] = ss_perturb[node2] + perturb_strength

    d_perturb = fs_perturb.copy()
    d_perturb[node2] = fs_perturb[node2] + perturb_strength

    m_fsfpa, _, fsfpa, stab2 = cut_find_fpas(theta_init=fs_perturb)
    m_ssfpa, _, ssfpa, stab3 = cut_find_fpas(theta_init=ss_perturb)
    m_dfpa, _, dfpa, stab4 = cut_find_fpas(theta_init=d_perturb)

    dists = {
        'fs': fsfpa - ifpa,
        'ss': ssfpa - ifpa,
        'd': dfpa - ifpa,
    }

    mod_dists = {
        'fs': m_fsfpa - m_ifpa,
        'ss': m_ssfpa - m_ifpa,
        'd': m_dfpa - m_ifpa,
    }

    total_stability = stab1 | stab2 | stab3 | stab4

    return dists, mod_dists, total_stability

def projection_distance(fpa1: np.ndarray, fpa2: np.ndarray, target_fpa: np.ndarray):
    A = np.column_stack((fpa1,fpa2))
    #calculate moore penrose pseudo inverses
    A_pinv = np.linalg.pinv(A)
    Proj_target = A @ A_pinv @ target_fpa

    Proj_distance = np.linalg.norm(target_fpa - Proj_target)
    return Proj_distance