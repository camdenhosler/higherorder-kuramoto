#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor

from src.oscillators import find_fpas

def run_trial(N,K,p_L,seed):

    G = nx.erdos_renyi_graph(n=N, p=p_L,seed=int(seed))
    A = nx.to_numpy_array(G)

    rng = np.random.default_rng(seed=seed)
    omega = rng.normal(loc=5.0, scale=1.0, size=N)

    t_start, t_end = 0.0, 30.0
    t_eval = np.linspace(t_start, t_end, 3000)

    theta_init = rng.uniform(low=0.0, high=2*np.pi, size=N)
    nx.set_node_attributes(G, dict(zip(G.nodes(), theta_init)), name="angle")

    sim_params = {
        'omega': omega,
        'K': K,
        'N': N,
        'A': A
    }

    theta, theta_deriv = find_fpas(t_start=t_start,t_end=t_end,theta_init=theta_init,t_eval=t_eval,params=sim_params)


    phase_init = theta[0]
    delta_theta = np.abs(theta - phase_init)

    return G, delta_theta
    
def run_K_trials(N,K_max,num_trials, p_L, seed):

    k = np.linspace(0.0, K_max, num_trials)
   
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_trial, N, k_i, p_L, seed)
            for k_i in k
        ]

        results_list = []
        for i, future in enumerate(futures):
            _, result = future.result()
            results_list.append(result)
            print(f"Trial {i+1}/{num_trials} complete")

        state_arr = np.column_stack(results_list)
        
        return state_arr, k

def figure1(G, delta_theta, N, p_L, K):
    pos = nx.spring_layout(G, seed=42)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    nx.draw_networkx_edges(G, pos, edge_color="slategrey", alpha=0.2,ax=ax1)

    colors = ['white', "#44bbe3"]
    custom_cmap = LinearSegmentedColormap.from_list("white_to_blue", colors)
    node_col = nx.draw_networkx_nodes(
    G, 
    pos,
    node_color=delta_theta, 
    cmap=custom_cmap, 
    vmin = 0,
    vmax = np.pi,
    node_size=50,
    edgecolors="black",
    linewidths=1,
    ax=ax1,
    )
    ax1.set_title(f"Erdős–Rényi Network (N = {N}, p = {p_L}, K = {K})")
    ax1.axis('off')

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

    plt.show()

def figure2(state_arr, k_arr):
    N, num_trials = state_arr.shape 
    k_flat = np.repeat(k_arr, N)         
    val_flat = state_arr.ravel(order="F")    

    num_k_bins = 15
    num_val_bins = 15

    counts, val_edges, k_edges = np.histogram2d(
        val_flat, k_flat,
        bins=[num_val_bins, num_k_bins],
        range=[[0, np.pi],[k_arr.min(), k_arr.max()]]
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=True)

    val_centers = (val_edges[:-1] + val_edges[1:]) / 2 
    k_centers = (k_edges[:-1] + k_edges[1:]) / 2  

    x_pos, y_pos = np.meshgrid(val_centers, k_centers, indexing="ij")
    x_pos = x_pos.ravel()
    y_pos = y_pos.ravel()
    z_pos = np.zeros_like(x_pos)

    dx = np.full_like(x_pos, val_edges[1] - val_edges[0])
    dy = np.full_like(y_pos, k_edges[1] - k_edges[0])
    dz = counts.ravel()

    mask = dz > 0
    x_pos = x_pos[mask]
    y_pos = y_pos[mask]
    z_pos = z_pos[mask]
    dx = dx[mask]
    dy = dy[mask]
    dz = dz[mask]  

    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color="#44bbe3", edgecolor='black', linewidth=0.8, shade=False)

    ax.view_init(elev=25, azim=-33)
    ax.set_box_aspect([1, 1, 0.6])
    fig.tight_layout()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi$"])
    ax.set_xlabel("Phase Difference from Node 0")
    ax.set_ylabel("Coupling Constant Epsilon")
    ax.set_zlabel("Number of Nodes")
    ax.set_title(f"Distribution of Relative Phase and Coupling Constant", y=0.97) 
    plt.show()

def main():
    N = 50
    K_max = 250
    num_trials = 100
    p_L = 0.2

    master_rng = np.random.default_rng()
    seed = master_rng.integers(0, 2**32 - 1, endpoint=True)

    G, fig1_dt = run_trial(N=N,K=100,p_L=p_L,seed=seed)
    figure1(G=G,delta_theta=fig1_dt,N=N,p_L=p_L,K=100)

    state_arr, k_arr = run_K_trials(N=N,K_max=K_max,num_trials=num_trials,p_L=p_L,seed=seed)
    figure2(state_arr=state_arr,k_arr=k_arr)

if __name__ == "__main__":
    main()
