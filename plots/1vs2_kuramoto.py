#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xgi
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

    omega = np.zeros(N)
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

    node1, node2 = rng.choice(np.arange(1, N), size=2, replace=False)
    h_dists, h_diff_dists, h_stability = perturbation(node1=node1,node2=node2,perturb_strength=perturbation_strength,
                                               t_start=t_start,t_end=t_end,theta_init=theta_init,params=high_params,func=hi.find_fpas)
    l_dists, l_diff_dists, l_stability = perturbation(node1=node1,node2=node2,perturb_strength=perturbation_strength,
                                               t_start=t_start,t_end=t_end,theta_init=theta_init,params=low_params,func=lo.find_fpas)

    if not h_stability:
        h_spanresult = np.nan
        h_orthresult = np.nan
    else:
        h_fs_diff = h_diff_dists['fs']
        h_ss_diff = h_diff_dists['ss']
        h_d_diff = h_diff_dists['d']
        h_orthresult, h_spanresult, h_deg = projection_distance(h_fs_diff,h_ss_diff,h_d_diff)
        if h_deg:
            h_spanresult = np.nan
            h_orthresult = np.nan

    if not l_stability:
        l_spanresult = np.nan
        l_orthresult = np.nan
    else:
        l_fs_diff = l_diff_dists['fs']
        l_ss_diff = l_diff_dists['ss']
        l_d_diff = l_diff_dists['d']
        l_orthresult, l_spanresult, l_deg = projection_distance(l_fs_diff,l_ss_diff,l_d_diff)
        if l_deg:
            l_spanresult = np.nan
            l_orthresult = np.nan

    #span is the difference from projected vector to the sum of the singles
    #orth is the difference from the span to the double
    return h_orthresult, l_orthresult

def calculate_distribution_parallel(num_trials, N, K, p_H, p_L, perturbation_strength, max_retries=40):
    ps = [0.0, p_H]
    t_start, t_end = 0.0, 10.0
    master_rng = np.random.default_rng()

    h_results = [np.nan] * num_trials
    l_results = [np.nan] * num_trials
    pending_indices = list(range(num_trials))
    attempt = 0

    with ProcessPoolExecutor() as executor:
        while pending_indices and attempt < max_retries:
            seeds = master_rng.integers(0, 2**32 - 1, size=len(pending_indices))
            futures = {
                idx: executor.submit(run_trial, seed, N, K, t_start, t_end, ps, p_L, perturbation_strength)
                for idx, seed in zip(pending_indices, seeds)
            }

            still_pending = []
            for idx, future in futures.items():
                h_val, l_val = future.result()
                h_results[idx] = h_val
                l_results[idx] = l_val
                if np.isnan(h_val) or np.isnan(l_val):
                    still_pending.append(idx)

            resolved = len(pending_indices) - len(still_pending)
            print(f"Attempt {attempt+1}: resolved {resolved}/{len(pending_indices)}, "
                  f"{len(still_pending)} still NaN")

            pending_indices = still_pending
            attempt += 1

    if pending_indices:
        print(f"WARNING: {len(pending_indices)} of {num_trials} trials never "
              f"converged after {max_retries} attempts and remain NaN in both arrays.")

    return h_results, l_results

def main():
    num_trials = 10000
    N = 20
    p_H = 0.03
    dp_L = p_H * (N - 2) * (1 / 2) 
    ep_L = p_H * (N - 2) * (1 / 3) 
    p_L = min(dp_L, 1.0)
    K = 100.0
    pert_str = np.pi / 2

    # Run simulations
    h_epis_arr, l_epis_arr = calculate_distribution_parallel(num_trials, N, K, p_H, p_L, pert_str)

    # Clean NaNs
    h_clean = np.asarray(h_epis_arr)[~np.isnan(h_epis_arr)]
    l_clean = np.asarray(l_epis_arr)[~np.isnan(l_epis_arr)]

    print(f"Unstable trials: higher-order={num_trials - len(h_clean)}, lower-order={num_trials - len(l_clean)}")

    # Moments
    h_mean, h_std = np.mean(h_clean), np.std(h_clean)
    l_mean, l_std = np.mean(l_clean), np.std(l_clean)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    # Colors: Blue (Higher-order), Vermilion (Lower-order)
    c_high = "#0072B2"
    c_low = "#D55E00"

    # Share bins explicitly across both datasets
    min_x = min(h_clean.min(), l_clean.min())
    max_x = max(h_clean.max(), l_clean.max())
    bins = np.linspace(min_x, max_x, 40)

    # Overlaid step histograms
    ax.hist(h_clean, bins=bins, color=c_high, density=True, histtype="stepfilled", alpha=0.15)
    ax.hist(h_clean, bins=bins, color=c_high, density=True, histtype="step", linewidth=1.5, 
            label=f"Higher-Order ($p_H = {p_H}$)")

    ax.hist(l_clean, bins=bins, color=c_low, density=True, histtype="stepfilled", alpha=0.15)
    ax.hist(l_clean, bins=bins, color=c_low, density=True, histtype="step", linewidth=1.5, 
            label=f"Lower-Order ($p_L = {p_L:.3f}$)")

    # Axis configuration
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)
    ax.set_xlabel("Epistasis", fontsize=10)
    ax.set_ylabel("Probability Density (Log Scale)", fontsize=10)

    # Clean spines and grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", which="both", top=False, right=False)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    ax.legend(frameon=False, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.show()

    # # Colors
    # c_high = "#0072B2"  # Blue
    # c_low = "#D55E00"   # Vermilion

    # # Linear bins across shared data range
    # min_x = min(h_clean.min(), l_clean.min())
    # max_x = max(h_clean.max(), l_clean.max())
    # bins = np.linspace(min_x, max_x, 35)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150, sharey=True)

    # # --- Panel 1: Higher-Order ---
    # ax1.hist(h_clean, bins=bins, color=c_high, alpha=0.3, density=True, label="Density")
    # ax1.hist(h_clean, bins=bins, color=c_high, histtype="step", linewidth=1.2, density=True)
    
    # ax1.axvline(h_mean, color=c_high, linestyle="--", linewidth=1.5, label=f"$\mu = {h_mean:.2e}$")
    # ax1.axvspan(h_mean - h_std, h_mean + h_std, color=c_high, alpha=0.12, label=f"$\sigma = {h_std:.2e}$")

    # ax1.set_yscale("log")
    # # Setting an explicit positive bottom limit prevents log(0) NaN issues
    # ax1.set_ylim(bottom=1e-3) 
    # ax1.set_xlabel("Epistasis", fontsize=10)
    # ax1.set_ylabel("Probability Density (Log Scale)", fontsize=10)
    # ax1.set_title(f"Higher-Order ($p_H = {p_H}$, $N = {N}$)", fontsize=11)
    # ax1.spines["top"].set_visible(False)
    # ax1.spines["right"].set_visible(False)
    # ax1.tick_params(direction="in", which="both", top=False, right=False)
    # ax1.grid(True, which="both", linestyle=":", alpha=0.4)
    # ax1.legend(frameon=False, fontsize=9)

    # # --- Panel 2: Lower-Order ---
    # ax2.hist(l_clean, bins=bins, color=c_low, alpha=0.3, density=True, label="Density")
    # ax2.hist(l_clean, bins=bins, color=c_low, histtype="step", linewidth=1.2, density=True)
    
    # ax2.axvline(l_mean, color=c_low, linestyle="--", linewidth=1.5, label=f"$\mu = {l_mean:.2e}$")
    # ax2.axvspan(l_mean - l_std, l_mean + l_std, color=c_low, alpha=0.12, label=f"$\sigma = {l_std:.2e}$")

    # ax2.set_yscale("log")
    # ax2.set_ylim(bottom=1e-3)
    # ax2.set_xlabel("Epistasis", fontsize=10)
    # ax2.set_title(f"Lower-Order ($p_L = {p_L:.3f}$, $N = {N}$)", fontsize=11)
    # ax2.spines["top"].set_visible(False)
    # ax2.spines["right"].set_visible(False)
    # ax2.tick_params(direction="in", which="both", top=False, right=False)
    # ax2.grid(True, which="both", linestyle=":", alpha=0.4)
    # ax2.legend(frameon=False, fontsize=9)

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()