from typing import Tuple, Dict, Any
from functools import partial
import numpy as np

from .oscillators import find_fpas
from .higher_oscillators import higher_find_fpas
#from ..scratchwork.cuthigher_oscillators import cuthigher_find_fpas

def perturbation(node1: int, node2: int,t_start: float, t_end: float, theta_init: np.ndarray, t_eval: np.ndarray,  params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    cut_find_fpas = partial(find_fpas,t_start=t_start,t_end=t_end,t_eval=t_eval,params=params)

    initial_state, _ = cut_find_fpas(theta_init=theta_init)

    perturb_strength = np.pi / 2

    fs_perturb = initial_state.copy()
    ss_perturb = initial_state.copy()

    fs_perturb[node1] = fs_perturb[node1] + perturb_strength
    ss_perturb[node2] = ss_perturb[node2] + perturb_strength

    d_perturb = fs_perturb.copy()
    d_perturb[node2] = fs_perturb[node2] + perturb_strength

    fsfpa, _ = cut_find_fpas(theta_init=fs_perturb)
    ssfpa, _ = cut_find_fpas(theta_init=ss_perturb)
    dfpa, _ = cut_find_fpas(theta_init=d_perturb)

    fsfpa_dist = fsfpa - initial_state
    ssfpa_dist = ssfpa - initial_state
    dfpa_dist = dfpa - initial_state


    return fsfpa_dist, ssfpa_dist, dfpa_dist

def higher_perturbation(node1: int, node2: int,t_start: float, t_end: float, theta_init: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    cut_find_fpas = partial(higher_find_fpas,t_start=t_start,t_end=t_end,params=params)

    initial_state, _ = cut_find_fpas(theta_init=theta_init)

    perturb_strength = np.pi / 2

    fs_perturb = initial_state.copy()
    ss_perturb = initial_state.copy()

    fs_perturb[node1] = fs_perturb[node1] + perturb_strength
    ss_perturb[node2] = ss_perturb[node2] + perturb_strength

    d_perturb = fs_perturb.copy()
    d_perturb[node2] = fs_perturb[node2] + perturb_strength
    
    fsfpa, _ = cut_find_fpas(theta_init=fs_perturb)
    ssfpa, _ = cut_find_fpas(theta_init=ss_perturb)
    dfpa, _ = cut_find_fpas(theta_init=d_perturb)

    fsfpa_dist = fsfpa - initial_state
    ssfpa_dist = ssfpa - initial_state
    dfpa_dist = dfpa - initial_state

    return fsfpa_dist, ssfpa_dist, dfpa_dist

# def cuthigher_perturbation(node1: int, node2: int,t_start: float, t_end: float, theta_init: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

#     cut_find_fpas = partial(cuthigher_find_fpas,t_start=t_start,t_end=t_end,params=params)

#     initial_state, _, t0 = cut_find_fpas(theta_init=theta_init)

#     perturb_strength = np.pi / 2

#     fs_perturb = initial_state.copy()
#     ss_perturb = initial_state.copy()

#     fs_perturb[node1] = fs_perturb[node1] + perturb_strength
#     ss_perturb[node2] = ss_perturb[node2] + perturb_strength

#     d_perturb = fs_perturb.copy()
#     d_perturb[node2] = fs_perturb[node2] + perturb_strength

#     fsfpa, _, t1 = cut_find_fpas(theta_init=fs_perturb)
#     ssfpa, _, t2 = cut_find_fpas(theta_init=ss_perturb)
#     dfpa, _, t3 = cut_find_fpas(theta_init=d_perturb)

#     fsfpa_dist = fsfpa - initial_state
#     ssfpa_dist = ssfpa - initial_state
#     dfpa_dist = dfpa - initial_state

#     t = max(t0,t1,t2,t3)

#     return fsfpa_dist, ssfpa_dist, dfpa_dist, t

def projection_distance(fpa1: np.ndarray, fpa2: np.ndarray, target_fpa: np.ndarray):
    A = np.column_stack((fpa1,fpa2))
    #calculate moore penrose pseudo inverses
    #run tests for this func
    A_pinv = np.linalg.pinv(A)
    Proj_target = A @ A_pinv @ target_fpa

    Proj_distance = np.linalg.norm(target_fpa - Proj_target)
    return Proj_distance

