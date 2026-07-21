#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xgi
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import moment

import src.higher_oscillators as hi
import src.lower_oscillators as lo
from src.perturbations import perturbation, projection_distance

def adjacency_tensor(H, order):
    N = H.num_nodes
    shape = tuple([N] * (order + 1))
    tensor = np.zeros(shape)

    edges = H.edges.filterby("order", order)
    for _, members in edges.members(dtype=dict).items():
        for idcs in permutations(members):
            tensor[idcs] = 1

    return tensor

def run_trial(seed,N,K,t_start,t_end,ps,p_L,perturbation_strength):
    rng = np.random.default_rng(seed)

    omega = rng.normal(loc=5.0, scale=1.0, size=N)
    theta_init = rng.uniform(low=0.0, high=2*np.pi, size=N)

    H = xgi.fast_random_hypergraph(N, ps)
    A_H = adjacency_tensor(H,2)
    h_sparse_A = hi.build_sparse_A(A_H)

    high_params = {
        'omega': omega,
        'K': K,
        'N': N,
        'sparse_A': h_sparse_A
    }

    G = nx.erdos_renyi_graph(N, p_L, seed=int(seed))
    A_L = nx.to_numpy_array(G)
    l_sparse_A = lo.build_sparse_A(A_L)

    low_params = {
        'omega': omega,
        'K': K,
        'N': N,
        'sparse_A': l_sparse_A
    }

    node1, node2 = rng.choice(N, size=2, replace=False)
    h_mod_dists, h_dists, h_stability = perturbation(node1=node1,node2=node2,perturb_strength=perturbation_strength,
                                               t_start=t_start,t_end=t_end,theta_init=theta_init,params=high_params,func=hi.find_fpas)
    l_mod_dists, l_dists, l_stability = perturbation(node1=node1,node2=node2,perturb_strength=perturbation_strength,
                                               t_start=t_start,t_end=t_end,theta_init=theta_init,params=low_params,func=lo.find_fpas)

    if not h_stability:
        h_result = np.nan
    else:
        h_fs_diff = h_mod_dists['fs']
        h_ss_diff = h_mod_dists['ss']
        h_d_diff = h_mod_dists['d']
        h_result = projection_distance(h_fs_diff,h_ss_diff,h_d_diff)

    if not l_stability:
        l_result = np.nan
    else:
        l_fs_diff = l_mod_dists['fs']
        l_ss_diff = l_mod_dists['ss']
        l_d_diff = l_mod_dists['d']
        l_result = projection_distance(l_fs_diff,l_ss_diff,l_d_diff)


    return h_result, l_result


def calculate_distribution_parallel(num_trials, N, K, p_H, p_L, perturbation_strength):
    
    ps = [0.0,p_H]
    
    t_start, t_end = 0.0, 10.0

    master_rng = np.random.default_rng()
    #checkout num trials vs N her
    seeds = master_rng.integers(0, 2**32 - 1, size=num_trials)
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_trial, seed, N, K, t_start, t_end, ps, p_L, perturbation_strength)
            for seed in seeds
        ]

        h_epis_arr = []
        l_epis_arr = []
        for i, future in enumerate(futures):
            h_epis_arr.append(future.result()[0])
            l_epis_arr.append(future.result()[1])
            print(f"Trial {i+1}/{num_trials} complete")
        
        return h_epis_arr, l_epis_arr

def main():
    num_trials = 1000
    N = 20
    p_H = 0.03
    np_L = p_H * (N - 2) * (2 / 9) #expected number of connected nodes
    ep_L = p_H * (N - 2) * (1 / 3) #expected edge number
    dp_L = p_H * (N - 2) * (1 / 2) #expected degree
    p_L = min(ep_L,1)
    K = 50.0
    pert_str = np.pi / 2
    h_epis_arr, l_epis_arr = calculate_distribution_parallel(num_trials,N,K,p_H,p_L,pert_str)

    h_epis_arr = np.array(h_epis_arr, dtype=float)
    l_epis_arr = np.array(l_epis_arr, dtype=float)

    n_h_unstable = np.isnan(h_epis_arr).sum()
    n_l_unstable = np.isnan(l_epis_arr).sum()
    print(f"Unstable trials: higher-order={n_h_unstable}, lower-order={n_l_unstable}")

    h_m1 = moment(h_epis_arr, order=1, center=0, nan_policy='omit')
    h_m2 = moment(h_epis_arr, order=2, nan_policy='omit')
    l_m1 = moment(l_epis_arr, order=1, center=0, nan_policy='omit')
    l_m2 = moment(l_epis_arr, order=2, nan_policy='omit')

    # --- shared x range ---
    shared_max_x = max(
        np.nanmax(h_epis_arr) if not np.all(np.isnan(h_epis_arr)) else 0,
        np.nanmax(l_epis_arr) if not np.all(np.isnan(l_epis_arr)) else 0,
    )

    sns.set_theme(style="whitegrid")
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bin_num = int(num_trials / 30)
    binrange = (0, shared_max_x)

    h_counts, _ = np.histogram(h_epis_arr[~np.isnan(h_epis_arr)], bins=bin_num, range=binrange)
    l_counts, _ = np.histogram(l_epis_arr[~np.isnan(l_epis_arr)], bins=bin_num, range=binrange)
    shared_max_y = max(h_counts.max(), l_counts.max())

    sns.histplot(
        h_epis_arr, 
        color="skyblue", 
        edgecolor="black",
        bins = bin_num,
        binrange=binrange,
        ax=ax1,
    )
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(0, shared_max_x*1.05)
    ax1.set_ylim(0, shared_max_y*1.05)
    ax1.set_xlabel("Epistasis")
    ax1.set_ylabel("Number of Trials")
    ax1.set_title(f"Epistasis in 2nd Order ER (p = {p_H}, N = {N}, K = {K})")
    ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    sns.histplot(
        l_epis_arr, 
        color="skyblue", 
        edgecolor="black",
        bins = bin_num,
        binrange=binrange,
        ax=ax2,
    )
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(0, shared_max_x*1.05)
    ax2.set_ylim(0, shared_max_y*1.05)
    ax2.set_xlabel("Epistasis")
    ax2.set_ylabel("Number of Trials")
    ax2.set_title(f"Epistasis in 1st Order ER (p = {p_L}, N = {N}, K = {K})")
    ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    plt.show()


if __name__ == "__main__":
    main()
