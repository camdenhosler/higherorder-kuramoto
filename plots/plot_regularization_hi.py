#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import xgi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from concurrent.futures import ProcessPoolExecutor

from collections import namedtuple
from juliacall import Main as jl

from src.sparsify import sparse_adjacency_tensor

def init_julia(parent_directory, julia_file_path):
    from juliacall import Main as jl
    jl.seval(f'import Pkg; Pkg.activate(raw"{parent_directory}")')
    jl.include(str(julia_file_path))

def run_trial(seed, N, ps, K):
   
    H = xgi.fast_random_hypergraph(int(N), ps, seed=int(seed))
    idx_i, idx_j, idx_k, vals = sparse_adjacency_tensor(H, 2)

    rng = np.random.default_rng(seed=seed)
    theta_init = rng.uniform(low=0.0, high=2*np.pi, size=N)

    DynParamsH = namedtuple('ModelParamsH', ['omega', 'K', 'N', 'idx_i', 'idx_j', 'idx_k', 'vals'])
    H_params = DynParamsH(
        omega=np.zeros(N),
        K=K,
        N=N,
        idx_i=idx_i + 1,
        idx_j=idx_j + 1,
        idx_k=idx_k + 1,
        vals=vals,
    )

    HigherEpisData = jl.evolve_to_fixed_point(theta_init, H_params)
    theta = HigherEpisData.State

    phase_init = theta[0]
    delta_theta = np.abs(np.array(theta) - phase_init)

    return H, delta_theta

def run_K_trials(num_trials, seeds, N, K_max, ps, parent_directory, julia_file_path):

    k = np.linspace(0.0, K_max, num_trials)
   
    with ProcessPoolExecutor(initializer=init_julia, initargs=(str(parent_directory), str(julia_file_path))) as executor:
        futures = [
            executor.submit(run_trial, seed=int(s), N=N, ps=ps, K=k_i)
            for s, k_i in zip(seeds, k)
        ]

        results_list = []
        for i, future in enumerate(futures):
            _, result = future.result()
            results_list.append(result)
            print(f"Trial {i+1}/{num_trials} complete")

        state_arr = np.column_stack(results_list)
        
        return state_arr, k

def figure1(H,delta_theta,N,ps,K): 

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    pos = xgi.barycenter_spring_layout(H)

    colors = ['white', "#44bbe3"]
    custom_cmap = LinearSegmentedColormap.from_list("white_to_blue", colors)

    _, node_col = xgi.draw_nodes(H,
             node_fc=delta_theta,
             node_fc_cmap=custom_cmap,
             vmin = 0,
             vmax = np.pi,
             ax=ax1,
             pos=pos,
             )
    print(xgi.is_connected(H))

    ax1.set_title(f"2nd Order Erdős–Rényi Network (N = {N}, p = {ps[1]}, K = {K})")

    cbar = plt.colorbar(node_col, label="Phase Difference from Node 0", location = "bottom")
    cbar.set_ticks([0, np.pi])
    cbar.set_ticklabels([r"$0$", r"$\pi$"])
    cbar.ax.yaxis.get_major_formatter().set_scientific(False)
    cbar.ax.yaxis.get_major_formatter().set_useOffset(False)

    num_bins = 20
    bin_width = np.pi / num_bins
    bins = np.linspace(0, np.pi + bin_width / 2, num_bins + 2)

    ax2.hist(delta_theta, bins=bins, color="#44bbe3", edgecolor="black")

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(0, None)
    ax2.set_xticks([0, np.pi])
    ax2.set_ylim(0, None)
    ax2.set_xticklabels([r"$0$", r"$\pi$"])
    ax2.set_xlabel("Phase Difference from Node 0")
    ax2.set_ylabel("Number of Nodes")
    ax2.set_title(f"Distribution of Relative Phase")
    ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)


def figure2(state_arr, k_arr):
    N, num_trials = state_arr.shape 
    k_flat = np.repeat(k_arr, N)         
    val_flat = state_arr.ravel(order="F")    

    num_k_bins = 15
    num_val_bins = 15

    val_binwidth = np.pi / num_val_bins

    H, val_edges, k_edges = np.histogram2d(
        val_flat, k_flat,
        bins=[num_val_bins, num_k_bins],
        range=[[0 - val_binwidth / 2, np.pi + val_binwidth / 2],[k_arr.min(), k_arr.max()]],
    )

    fig, ax = plt.subplots(figsize=(5,4), dpi=200)

    val_centers = (val_edges[:-1] + val_edges[1:]) / 2 
    k_centers = (k_edges[:-1] + k_edges[1:]) / 2  

    H_masked = H.copy()
    H_masked[H_masked == 0] = np.nan

    colors = ['white', "#44bbe3"]
    custom_cmap = LinearSegmentedColormap.from_list("white_to_blue", colors)

    ax.set_xticks([0, np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi$"])

    plt.pcolormesh(val_centers, k_centers, H_masked.T, cmap='viridis', shading='auto', norm = LogNorm())

    plt.colorbar(label='Counts per bin')
    plt.xlabel('Phase Difference From Node 0')
    plt.ylabel('Coupling Constant')
    plt.xlim(0 - val_binwidth / 2, val_binwidth / 2 + np.pi)
    plt.ylim(0, None)
    plt.title("Final state after 10,000 time steps")

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(script_directory)

    jl.seval(f'import Pkg; Pkg.activate(raw"{parent_directory}")')


    script_dir = Path(__file__).parent
    julia_file_path = script_dir.parent / "src" / "dynamics.jl"

    jl.include(str(julia_file_path))

    N = 50
    K_max = 1000
    num_trials = 100
    p_H = 0.03
    ps = [0.0,p_H]

    master_rng = np.random.default_rng()
    seed = master_rng.integers(0, 2**32 - 1, endpoint=True)
    seeds = master_rng.integers(0, 2**32 - 1, endpoint=True, size=num_trials)

    H, delta_theta = run_trial(N=N,ps=ps,K=100,seed=seed)
    state_arr, k_arr = run_K_trials(
        num_trials=num_trials,
        seeds=seeds, 
        N=N,
        K_max=K_max,
        ps=ps,
        parent_directory=parent_directory,
        julia_file_path=julia_file_path)

    figure1(H=H,delta_theta=delta_theta,N=N,ps=ps,K=100)
    figure2(state_arr=state_arr,k_arr=k_arr)

    plt.show()

if __name__ == "__main__":

    main()