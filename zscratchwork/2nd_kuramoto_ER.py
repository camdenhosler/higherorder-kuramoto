#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xgi
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import moment

import src.higher_oscillators as hi
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

def run_trial(seed,N,K,t_start,t_end,ps,perturbation_strength):
    H = xgi.fast_random_hypergraph(N, ps)
    A = adjacency_tensor(H,2)
    sparse_A = hi.build_sparse_A(A)

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
    mod_dists, dists, stability = perturbation(node1=node1,node2=node2,perturb_strength=perturbation_strength,
                                               t_start=t_start,t_end=t_end,theta_init=theta_init,params=sim_params,func=hi.find_fpas)

    if not stability:
        return -1
    
    fs_diff = mod_dists['fs']
    ss_diff = mod_dists['ss']
    d_diff = mod_dists['d']

    return projection_distance(fs_diff,ss_diff,d_diff)


def calculate_distribution_parallel(num_trials, N, p,perturbation_strength):
    
    ps = [0.0,p]
    K = 10000.0
    t_start, t_end = 0.0, 10.0

    master_rng = np.random.default_rng()
    #checkout num trials vs N her
    seeds = master_rng.integers(0, 2**32 - 1, size=num_trials)
    print(seeds)
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_trial, seed, N, K, t_start, t_end, ps, perturbation_strength)
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
    p = 0.03
    pert_str = np.pi / 2
    epis_arr = calculate_distribution_parallel(num_trials,N,p,pert_str)

    m1 = moment(epis_arr, order=1, center=0)
    m2 = moment(epis_arr, order=2)
    m3 = moment(epis_arr, order=3)
    m4 = moment(epis_arr, order=4)

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
        bins = bin_num,
    )
    plt.xlim(None, None)
    plt.xlabel("Epistasis in Higher Order Kuramoto")
    plt.ylabel("Number of trials")
    plt.title(f"Distribution over {num_trials} trials on Erdos Reyni {N} Node Network(p={p})")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
