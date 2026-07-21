#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import moment
from concurrent.futures import ProcessPoolExecutor

from src.perturbations import perturbation, projection_distance
import src.lower_oscillators as lo

def run_trial(seed,N,K,t_start,t_end, p, perturb_strength):
    G = nx.erdos_renyi_graph(N, p, seed=int(seed))
    A = nx.to_numpy_array(G)
    sparse_A = lo.build_sparse_A(A)

    rng = np.random.default_rng(seed)

    omega = rng.normal(loc=5.0, scale=1.0, size=N)
    theta_init = rng.uniform(low=0.0, high=2*np.pi, size=N)
    sim_params = {
        'omega': omega,
        'K': K,
        'N': N,
        'sparse_A': sparse_A
    }

    node1, node2 = rng.choice(N, size=2, replace=False)
    mod_dists, dists, stability = perturbation(node1=node1,node2=node2,perturb_strength=perturb_strength,
                                      t_start=t_start,t_end=t_end,theta_init=theta_init, params=sim_params, func=lo.find_fpas)

    if not stability:
        return -1
    
    fs_diff = mod_dists['fs']
    ss_diff = mod_dists['ss']
    d_diff = mod_dists['d']
    
    return projection_distance(fs_diff,ss_diff,d_diff)

def calculate_distribution(num_trials, N, p, perturb_strength):
    K = 10000.0
    t_start, t_end = 0.0, 10.0

    master_rng = np.random.default_rng()
    seeds = master_rng.integers(0, 2**32 - 1, size=num_trials)
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_trial, seed, N, K, t_start, t_end, p, perturb_strength)
            for seed in seeds
        ]

        epis_arr = []
        for i, future in enumerate(futures):
            epis_arr.append(future.result())
            print(f"Trial {i+1}/{num_trials} complete")
        
        return epis_arr


def main():
    num_trials = 1000
    N = 20
    p_H = 0.03
    pert_str = np.pi / 2

    np_L = p_H * (N - 2) * (2 / 9) #expected number of connected nodes
    ep_L = p_H * (N - 2) * (1 / 3) #expected edge number
    dp_L = p_H * (N - 2) * (1 / 2) #expected degree
    p = 0.7

    epis_arr = calculate_distribution(num_trials, N, p, pert_str)

    m1 = moment(epis_arr, order=1, center=0)
    m2 = moment(epis_arr, order=2)
    m3 = moment(epis_arr, order=3)
    m4 = moment(epis_arr, order=4)

    print(p)
    print(f"1st Central Moment (mean): {m1:.4f}")
    print(f"2nd Central Moment (Variance): {m2:.4f}")
    print(f"3rd Central Moment (Skewness precursor): {m3:.4f}")
    print(f"4th Central Moment (Kurtosis precursor): {m4:.4f}") 

    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(8, 5))
    bin_num = int(num_trials / 30)

    sns.histplot(
        epis_arr, 
        color="skyblue", 
        edgecolor="black",
        bins = bin_num 
    )
    plt.xlim(None, None)
    plt.xlabel("Epistasis in Lower Order Kuramoto")
    plt.ylabel("Number of trials")
    plt.title(f"Distribution over {num_trials} trials in Erdos Reyni {N} Node Network(p={p})")
    plt.grid(axis='y', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()

