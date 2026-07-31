#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import xgi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from collections import namedtuple
from juliacall import Main as jl

from src.sparsify import sparse_adjacency_tensor

script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)

jl.seval(f'import Pkg; Pkg.activate(raw"{parent_directory}")')


script_dir = Path(__file__).parent
julia_file_path1 = script_dir.parent / "src" / "dynamics.jl"
julia_file_path2 = script_dir.parent / "src" / "regularization.jl"

jl.include(str(julia_file_path1))
jl.include(str(julia_file_path2))

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

    DynParamsH = namedtuple('DynParamsH', ['omega', 'K', 'N', 'idx_i', 'idx_j', 'idx_k', 'vals'])
    H_params = DynParamsH(
        omega=np.zeros(N),
        K=K,
        N=N,
        idx_i=idx_i + 1,
        idx_j=idx_j + 1,
        idx_k=idx_k + 1,
        vals=vals,
    )

    HigherEpisData = jl.evolve_to_hkur_fixed_point(theta_init, H_params)
    theta = HigherEpisData.State

    phase_init = theta[0]
    delta_theta = np.abs(np.array(theta) - phase_init)

    return H, delta_theta

def calculate_dyadic_effect(N, ps, K_max, n_trials_K, n_trials_norm, master_rng: np.random.Generator | None = None):
    if master_rng is None:
        master_rng = np.random.default_rng()

    seeds = master_rng.bit_generator.seed_seq.spawn(n_trials_norm)

    K_batch = np.linspace(0.1, K_max, n_trials_K + 1)
    theta_init_chunks = []
    i_chunks_H, j_chunks_H, k_chunks_H, val_chunks_H = [], [], [], []
    nnz_per_trial_H = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        theta_init = rng.uniform(low=0.0, high=2*np.pi, size=N)

        theta_init_chunks.append(theta_init)

        seed_H = rng.integers(0, 2**32 - 1)

        H = xgi.fast_random_hypergraph(N, ps, seed=int(seed_H))
        idx_i_H, idx_j_H, idx_k_H, vals_H = sparse_adjacency_tensor(H, 2)

        i_chunks_H.append(idx_i_H)
        j_chunks_H.append(idx_j_H)
        k_chunks_H.append(idx_k_H)
        val_chunks_H.append(vals_H)
        nnz_per_trial_H.append(len(vals_H))

    theta_init_batch = np.stack(theta_init_chunks)

    idx_i_H_flat = np.concatenate(i_chunks_H)
    idx_j_H_flat = np.concatenate(j_chunks_H)
    idx_k_H_flat = np.concatenate(k_chunks_H)
    vals_H_flat  = np.concatenate(val_chunks_H)

    offsets_H = np.zeros(n_trials_norm + 1, dtype=np.int64)
    offsets_H[1:] = np.cumsum(nnz_per_trial_H)

    regDyParamsH = namedtuple('regDyParamsH', ['omega', 'K', 'N', 'idx_i', 'idx_j', 'idx_k', 'vals', 'offsets'])
    reg_dy_params = regDyParamsH(
        omega=np.zeros(N),
        K=K_batch,
        N=N,
        idx_i=idx_i_H_flat + 1,
        idx_j=idx_j_H_flat + 1,
        idx_k=idx_k_H_flat + 1,
        vals=vals_H_flat,
        offsets=offsets_H
    )

    averaged_diff_norm_vec = jl.calc_dyadic_domination(theta_init_batch, reg_dy_params, n_trials_norm)
    return averaged_diff_norm_vec, K_batch

def figure1(H,delta_theta,N,ps,K): 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
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

    cbar = fig.colorbar(
        node_col,
        ax=ax1,
        orientation="horizontal",
        pad=0.03,
        fraction=0.06,
    )
    cbar.set_label(r"$|\theta_i-\theta_0|$")
    cbar.set_ticks([0, np.pi])
    cbar.set_ticklabels([r"$0$", r"$\pi$"])
    cbar.ax.yaxis.get_major_formatter().set_scientific(False)
    cbar.ax.yaxis.get_major_formatter().set_useOffset(False)

    ax1.text(-0.05, 1.02, "(a)", transform=ax1.transAxes,
             fontweight="bold")

    num_bins = 20
    bin_width = np.pi / num_bins
    bins = np.linspace(0 - bin_width / 2, np.pi + bin_width / 2, num_bins + 2)

    ax2.hist(delta_theta, bins=bins, color="#44bbe3", edgecolor="black", histtype="stepfilled", alpha=0.8)

    ax2.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
    )
    ax2.set_xlim(None, None)
    ax2.set_xticks([0, np.pi])
    ax2.set_ylim(0, None)
    ax2.set_xticklabels([r"$0$", r"$\pi$"])
    ax2.set_xlabel("Phase Difference from Node 0")
    ax2.set_ylabel("Number of Nodes")
    ax2.text(-0.05, 1.02, "(b)",
             transform=ax2.transAxes,
             fontweight="bold")
    ax2.set_title(f"Distribution of Relative Phase")
    ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

def plot_dyadic_domination(averaged_diff_norm_vec, K_vec):
    
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    
    ax.plot(
        K_vec, 
        averaged_diff_norm_vec, 
        color="#2b7bba", 
        linewidth=1.5, 
        linestyle="-", 
        zorder=2
    )
    ax.scatter(
        K_vec, 
        averaged_diff_norm_vec, 
        color="#44bbe3", 
        edgecolor="black", 
        linewidths=0.6,
        s=25,
        alpha=0.9,
        zorder=3,
        label="Simulated Mean"
    )
    
    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True
    )
    
    ax.set_xlabel(r"Coupling Constant ($K$)")
    ax.set_ylabel(r"Averaged Norm Difference $\langle \|\Delta \boldsymbol{\theta}\| \rangle$")
    ax.set_title("Dyadic Domination Effect", pad=8)
    
    ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
    
    ax.margins(x=0.03, y=0.05)
    
    fig.tight_layout()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "dyadic_domination.png", format="png", dpi=300, bbox_inches="tight")


def main():
    set_figure_format()

    N = 20
    p_H = 0.03
    ps = [0.0,p_H]

    master_rng = np.random.default_rng()
    seed = master_rng.integers(0, 2**32 - 1, endpoint=True)

    H, delta_theta = run_trial(N=N,ps=ps,K=100,seed=seed)
    figure1(H=H,delta_theta=delta_theta,N=N,ps=ps,K=100)

    n_trials_K = 30
    n_trials_norm = 30
    K_max = 150

    averaged_diff_norm_vec, K_vec = calculate_dyadic_effect(N=N, ps=ps, K_max=K_max, n_trials_K=n_trials_K, n_trials_norm=n_trials_norm)

    plot_dyadic_domination(averaged_diff_norm_vec=averaged_diff_norm_vec, K_vec=K_vec)

    plt.tight_layout()
    plt.show()

def set_figure_format():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 10,
    })

if __name__ == "__main__":

    main()