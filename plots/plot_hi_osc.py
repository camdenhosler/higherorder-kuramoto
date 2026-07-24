#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xgi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from itertools import permutations
from concurrent.futures import ProcessPoolExecutor

from src.higher_oscillators import find_fpas, build_sparse_A

def adjacency_tensor(H, order):
    N = H.num_nodes
    shape = tuple([N] * (order + 1))
    tensor = np.zeros(shape)

    edges = H.edges.filterby("order", order)
    for _, members in edges.members(dtype=dict).items():
        for idcs in permutations(members):
            tensor[idcs] = 1

    return tensor

def run_trial(N,K,ps,seed):

    H = xgi.fast_random_hypergraph(N, ps, seed=int(seed))
    A = adjacency_tensor(H,2)
    sparse_A = build_sparse_A(A)
   
    rng = np.random.default_rng(seed=seed)
    omega = np.zeros(N)

    t_start, t_end = 0.0, 10.0

    theta_init = rng.uniform(low=0.0, high=2*np.pi, size=N)

    sim_params = {
        'omega': omega,
        'K': K,
        'N': N,
        'sparse_A': sparse_A
    }

    theta, theta_deriv, _, _, _= find_fpas(t_start=t_start,t_end=t_end,theta_init=theta_init,params=sim_params)

    phase_init = theta[0]
    delta_theta = np.abs(theta - phase_init)

    return H, delta_theta

def run_K_trials(N,K_max,num_trials, ps, seed):

    k = np.linspace(0.0, K_max, num_trials)
   
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_trial, N, k_i, ps, seed)
            for k_i in k
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
    N = 50
    K_max = 1000
    num_trials = 100
    p_H = 0.03
    ps = [0.0,p_H]

    master_rng = np.random.default_rng()
    seed = master_rng.integers(0, 2**32 - 1, endpoint=True)

    H, delta_theta = run_trial(N=N,ps=ps,K=100,seed=seed)
    state_arr, k_arr = run_K_trials(N=N,K_max=K_max,num_trials=num_trials,ps=ps,seed=seed)

    figure1(H=H,delta_theta=delta_theta,N=N,ps=ps,K=100)
    figure2(state_arr=state_arr,k_arr=k_arr)

    plt.show()

if __name__ == "__main__":
    main()



    # x_pos, y_pos = np.meshgrid(val_centers, k_centers, indexing="ij")
    # x_pos = x_pos.ravel()
    # y_pos = y_pos.ravel()
    # z_pos = np.zeros_like(x_pos)

    # dx = np.full_like(x_pos, val_edges[1] - val_edges[0])
    # dy = np.full_like(y_pos, k_edges[1] - k_edges[0])
    # dz = counts.ravel()

    # mask = dz > 0
    # x_pos = x_pos[mask]
    # y_pos = y_pos[mask]
    # z_pos = z_pos[mask]
    # dx = dx[mask]
    # dy = dy[mask]
    # dz = dz[mask]  

    # ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color="#44bbe3", edgecolor='black', linewidth=0.8, shade=False)

    # ax.view_init(elev=25, azim=-33)
    # ax.set_box_aspect([1, 1, 0.6])
    # fig.tight_layout()
    # ax.grid(True, alpha=0.3)
    # ax.set_xticks([0, np.pi])
    # ax.set_xticklabels([r"$0$", r"$\pi$"])
    # ax.set_xlabel("Phase Difference from Node 0")
    # ax.set_ylabel("Coupling Constant Epsilon")
    # ax.set_zlabel("Number of Nodes")
    # ax.set_title(f"Distribution of Relative Phase and Coupling Constant", y=0.97) 