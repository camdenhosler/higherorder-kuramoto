#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from collections import namedtuple

import numpy as np
import xgi
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

from juliacall import Main as jl

script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)

jl.seval(f'import Pkg; Pkg.activate(raw"{parent_directory}")')


script_dir = Path(__file__).parent
julia_file_path = script_dir.parent / "src" / "perturbations.jl"


jl.include(str(julia_file_path))

from src.sparsify import sparse_adjacency_matrix, sparse_adjacency_tensor

def generate_tensors(n_trials, N, p_H, p_L, seeds: list | None = None):
    if seeds is None:
        master_rng = np.random.default_rng()
        seeds = master_rng.bit_generator.seed_seq.spawn(n_trials)

    node1_list, node2_list, theta_init_chunks = [], [], []
    i_chunks_G, j_chunks_G, val_chunks_G = [], [], []
    i_chunks_H, j_chunks_H, k_chunks_H, val_chunks_H = [], [], [], []
    nnz_per_trial_G = []
    nnz_per_trial_H = []

    #each index is probability for that order edge
    ps_H = [0,p_H]
    #exclude our reference node, node 0
    #node_pool = np.arange(1, N)
    node_pool = np.arange(0, N)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        theta_init = rng.uniform(low=0.0, high=2*np.pi, size=N)
        node1, node2 = rng.choice(node_pool, size=2, replace=False) + 1

        node1_list.append(node1)
        node2_list.append(node2)
        theta_init_chunks.append(theta_init)

        seed_G = rng.integers(0, 2**32 - 1)
        seed_H = rng.integers(0, 2**32 - 1)

        G = nx.fast_gnp_random_graph(N, p_L, seed=int(seed_G))
        idx_i_G, idx_j_G, vals_G = sparse_adjacency_matrix(G)

        i_chunks_G.append(idx_i_G)
        j_chunks_G.append(idx_j_G)
        val_chunks_G.append(vals_G)
        nnz_per_trial_G.append(len(vals_G))

        H = xgi.fast_random_hypergraph(N, ps_H, seed=int(seed_H))
        idx_i_H, idx_j_H, idx_k_H, vals_H = sparse_adjacency_tensor(H, 2)

        i_chunks_H.append(idx_i_H)
        j_chunks_H.append(idx_j_H)
        k_chunks_H.append(idx_k_H)
        val_chunks_H.append(vals_H)
        nnz_per_trial_H.append(len(vals_H))

    theta_init_batch = np.stack(theta_init_chunks)   
    node1_batch = np.array(node1_list, dtype=np.int64)
    node2_batch = np.array(node2_list, dtype=np.int64)

    idx_i_G_flat = np.concatenate(i_chunks_G)
    idx_j_G_flat = np.concatenate(j_chunks_G)
    vals_G_flat  = np.concatenate(val_chunks_G)

    offsets_G = np.zeros(n_trials + 1, dtype=np.int64)
    offsets_G[1:] = np.cumsum(nnz_per_trial_G)

    idx_i_H_flat = np.concatenate(i_chunks_H)
    idx_j_H_flat = np.concatenate(j_chunks_H)
    idx_k_H_flat = np.concatenate(k_chunks_H)
    vals_H_flat  = np.concatenate(val_chunks_H)

    offsets_H = np.zeros(n_trials + 1, dtype=np.int64)
    offsets_H[1:] = np.cumsum(nnz_per_trial_H)

    return {
        "theta_init": theta_init_batch,
        "node1": node1_batch,
        "node2": node2_batch,

        "idx_i_G": idx_i_G_flat,
        "idx_j_G": idx_j_G_flat,
        "vals_G": vals_G_flat,
        "offsets_G": offsets_G,

        "idx_i_H": idx_i_H_flat,
        "idx_j_H": idx_j_H_flat,
        "idx_k_H": idx_k_H_flat,
        "vals_H": vals_H_flat,
        "offsets_H": offsets_H,
    }


def calculate_epistasis(n_trials, N, p_H, p_L, K, pert_str, seeds: list | None = None):
    
    batches = generate_tensors(n_trials,N,p_H,p_L, seeds)

    ModelParamsH = namedtuple('ModelParamsH', ['omega', 'K', 'N', 'idx_i', 'idx_j', 'idx_k', 'vals', 'offsets', 'pert_str'])
    ModelParamsG = namedtuple('ModelParamsG', ['omega', 'K', 'N', 'idx_i', 'idx_j', 'vals', 'offsets', 'pert_str'])

    #indices must be shifted since julia use index 1 to begin indexing
    H_params = ModelParamsH(
        omega=np.zeros(N),
        K=K,
        N=N,
        idx_i=batches['idx_i_H'] + 1,
        idx_j=batches['idx_j_H'] + 1,
        idx_k=batches['idx_k_H'] + 1,
        vals=batches['vals_H'],
        offsets=batches['offsets_H'],
        pert_str=pert_str,
    )

    G_params = ModelParamsG(
        omega=np.zeros(N),
        K=K,
        N=N,
        idx_i=batches['idx_i_G'] + 1,
        idx_j=batches['idx_j_G'] + 1,
        vals=batches['vals_G'],
        offsets=batches['offsets_G'],
        pert_str=pert_str,
    )

    HigherEpisData = jl.calc_h_epistasis(batches['theta_init'], batches['node1'], batches['node2'], H_params)
    LowerEpisData = jl.calc_l_epistasis(batches['theta_init'], batches['node1'], batches['node2'], G_params)

    rel_higherL1 = np.asarray(HigherEpisData.RelL1).ravel()
    rel_lowerL1 = np.asarray(LowerEpisData.RelL1).ravel()
    
    higherL1 = np.asarray(HigherEpisData.L1).ravel()
    lowerL1 = np.asarray(LowerEpisData.L1).ravel()

    higherL2 = np.asarray(HigherEpisData.L2).ravel()
    lowerL2 = np.asarray(LowerEpisData.L2).ravel()

    EpisArr = namedtuple('EpisArr', ['RelHL1', 'RelLL1', 'HL1', 'LL1', 'HL2', 'LL2'])

    return EpisArr(
        RelHL1=rel_higherL1,
        RelLL1=rel_lowerL1,
        HL1=higherL1,
        LL1=lowerL1,
        HL2=higherL2,
        LL2=lowerL2,
    )


def plot_epistasis(higher, lower, N, p_H, p_L, K, pert_str):
    higher = higher[np.isfinite(higher)]
    lower = lower[np.isfinite(lower)]

    cutoff = min(len(higher), len(lower))
    higher_cutoff = higher[:cutoff]
    lower_cutoff  = lower[:cutoff]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6))

    # --- Hist ---
    ax1 = axes[0]
    bins = np.linspace(0, max(higher.max(), lower.max()), 35)

    ax1.hist(higher_cutoff, bins=bins, alpha=0.4, color='#c0392b',
            label=f'Higher ($p_H = {p_H}$)', histtype='stepfilled', linewidth=1.5)
    ax1.hist(lower_cutoff,  bins=bins, alpha=0.4, color='#2980b9',
            label=f'Lower ($p_L = {p_L}$)', histtype='stepfilled', linewidth=1.5)
    ax1.set_xlabel(r'$d_{\mathrm{nonrel}}$')
    ax1.set_ylabel('Count')
    ax1.tick_params(direction='in', which='both', top=True, right=True)
    ax1.legend(frameon=False, loc='upper right')

    ax1.text(0.58, 0.70, f'$N={N}, K={K}$', transform=ax1.transAxes, fontsize=8)
    ax1.text(-0.05, 1.02, '(a)', transform=ax1.transAxes, fontweight='bold')

    # --- CDF ---
    ax2 = axes[1]
    for data, color, label in [(higher, '#c0392b', f'Higher ($p_H = {p_H}$)'),
                            (lower, '#2980b9', f'Lower ($p_L = {p_L}$)')]:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax2.plot(sorted_data, cdf, color=color, linewidth=1.2, label=label)

    ax2.set_ylim(1e-3, 1.1)
    ax2.set_xlabel(r'$d_{\mathrm{nonrel}}$')
    ax2.set_ylabel(r'$F(d_{\mathrm{nonrel}})$')
    ax2.tick_params(direction='in', which='both', top=True, right=True)
    ax2.legend(frameon=False, loc='lower right')
    ax2.text(-0.05, 1.02, '(b)', transform=ax2.transAxes, fontweight='bold')
    ax2.text(0.58, 0.275, f'$N={N}, K={K}$', transform=ax2.transAxes, fontsize=8)

    plt.tight_layout()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'kuramoto_epis.png', format='png', dpi=300, bbox_inches='tight')  


def plot_minimum_hi_epistasis_graphs(higher, seeds, N, p_H, K, pert_str):
    ps_H = [0.0, p_H]
    node_pool = np.arange(0, N)

    min_idx = np.nanargmin(higher)
    min_master_seed = seeds[min_idx]

    min_rng = np.random.default_rng(min_master_seed)

    min_theta_init = min_rng.uniform(low=0.0, high=2*np.pi, size=N)
    min_node1, min_node2 = min_rng.choice(node_pool, size=2, replace=False)
    
    _ = min_rng.integers(0, 2**32 - 1)
    min_seed = min_rng.integers(0, 2**32 - 1)


    max_idx = np.nanargmax(higher)
    max_master_seed = seeds[max_idx]

    max_rng = np.random.default_rng(max_master_seed)

    max_theta_init = max_rng.uniform(low=0.0, high=2*np.pi, size=N)
    max_node1, max_node2 = max_rng.choice(node_pool, size=2, replace=False)
    
    _ = max_rng.integers(0, 2**32 - 1)
    max_seed = max_rng.integers(0, 2**32 - 1)


    min_H = xgi.fast_random_hypergraph(N, ps_H, seed=int(min_seed))

    min_perturbed_nodes_arr = np.zeros(N)
    min_perturbed_nodes_arr[min_node1] = 1
    min_perturbed_nodes_arr[min_node2] = 1
    
    min_idx_i, min_idx_j, min_idx_k, min_vals = sparse_adjacency_tensor(min_H, 2)

    max_H = xgi.fast_random_hypergraph(N, ps_H, seed=int(max_seed))

    max_perturbed_nodes_arr = np.zeros(N)
    max_perturbed_nodes_arr[max_node1] = 1
    max_perturbed_nodes_arr[max_node2] = 1
    
    max_idx_i, max_idx_j, max_idx_k, max_vals = sparse_adjacency_tensor(max_H, 2)

    PertParamsH = namedtuple('PertParamsH', ['omega', 'K', 'N', 'idx_i', 'idx_j', 'idx_k', 'vals', 'pert_str'])

    min_H_params = PertParamsH(
        omega=np.zeros(N),
        K=K,
        N=N,
        idx_i=min_idx_i + 1,
        idx_j=min_idx_j + 1,
        idx_k=min_idx_k + 1,
        vals=min_vals,
        pert_str=pert_str,
    )

    max_H_params = PertParamsH(
        omega=np.zeros(N),
        K=K,
        N=N,
        idx_i=max_idx_i + 1,
        idx_j=max_idx_j + 1,
        idx_k=max_idx_k + 1,
        vals=max_vals,
        pert_str=pert_str,
    )

    MinHigherEpisData = jl.perturbation_hkur(min_node1 + 1, min_node2 + 1, min_theta_init, min_H_params, return_initial=True)
    MaxHigherEpisData = jl.perturbation_hkur(max_node1 + 1, max_node2 + 1, max_theta_init, max_H_params, return_initial=True)

    MinStateVectors = MinHigherEpisData.StateVectors
    MaxStateVectors = MaxHigherEpisData.StateVectors

    min_pert_nodes_set = {min_node1, min_node2}
    max_pert_nodes_set = {max_node1, max_node2}

    min_connected_edge_ids = set().union(*(min_H.nodes.memberships(v) for v in min_pert_nodes_set))
    min_H_sub = xgi.subhypergraph(min_H, edges=min_connected_edge_ids)
    max_connected_edge_ids = set().union(*(max_H.nodes.memberships(v) for v in max_pert_nodes_set))
    max_H_sub = xgi.subhypergraph(max_H, edges=max_connected_edge_ids)

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.2), gridspec_kw={'width_ratios': [1, 1.5]})

    ax1 = axes[0,0]

    min_pos = xgi.barycenter_spring_layout(min_H, k=1.5)

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("binary")

    min_edge_colors = cmap(norm(min_perturbed_nodes_arr))

    _, node_col = xgi.draw_nodes(
        min_H,
        node_fc=min_theta_init,
        node_fc_cmap='viridis',
        node_ec=min_edge_colors,
        vmin = 0,
        vmax = 2 * np.pi,
        node_size=8,
        node_lw=2,
        ax=ax1,
        pos=min_pos,
        )
    
    xgi.draw_hyperedges(min_H_sub, pos=min_pos, ax=ax1, alpha=0.1)

    cbar1 = fig.colorbar(
        node_col,
        ax=ax1,
        orientation="horizontal",
        pad=0.03,
        fraction=0.06,
    )
    cbar1.set_label(r"Initial $\theta$")
    cbar1.set_ticks([0, 2 * np.pi])
    cbar1.set_ticklabels([r"$0$", r"$2 \pi$"])
    cbar1.ax.yaxis.get_major_formatter().set_scientific(False)
    cbar1.ax.yaxis.get_major_formatter().set_useOffset(False)
    ax1.text(-0.05, 1.02, '(a)', transform=ax1.transAxes, fontweight='bold')
    ax1.text(0.60, 0.89, f'$N={N}, K={K}$', transform=ax1.transAxes, fontsize=8)

    ax1.set_axis_on()

    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = axes[0,1]

    min_init = np.array(MinStateVectors.I)
    min_s1 = np.array(MinStateVectors.S1)
    min_s2 = np.array(MinStateVectors.S2)
    min_ex = min_s1 + min_s2
    min_d = np.array(MinStateVectors.D)

    data = np.vstack([
        min_init % (2 * np.pi),
        min_s1 % (2 * np.pi),
        min_s2 % (2 * np.pi),
        min_ex % (2 * np.pi),
        min_d % (2 * np.pi)
        ])
    
    im = ax2.imshow(data, cmap="viridis", aspect="auto", interpolation="nearest", vmin=0, vmax=2*np.pi)
    
    cbar2 = fig.colorbar(im, ax=ax2, pad=0.02, fraction=0.046)
    cbar2.ax.tick_params(labelsize=8, direction="in")
    cbar2.set_label(r"Final $\theta$", fontsize=9)
    cbar2.minorticks_on()
    
    row_labels = [r"$Initial$",r"$Single 1$", r"$Single 2$", r"$Expected$", r"$Double$"]
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(row_labels, fontsize=8)
    
    ax2.set_xticks(np.arange(0, 20, 2))  # Tick every 2 units for clean look
    ax2.set_xticks(np.arange(0, 20, 1), minor=True)
    ax2.set_xlabel("Node States", fontsize=9)
    
    ax2.text(-0.05, 1.02, '(b)', transform=ax2.transAxes, fontweight='bold')
    ax2.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)

    ax3 = axes[1,0]

    max_pos = xgi.barycenter_spring_layout(max_H, k=1.5)

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("binary")

    max_edge_colors = cmap(norm(max_perturbed_nodes_arr))

    _, node_col = xgi.draw_nodes(
        max_H,
        node_fc=max_theta_init,
        node_fc_cmap='viridis',
        node_ec=max_edge_colors,
        vmin = 0,
        vmax = 2 * np.pi,
        node_size=8,
        node_lw=2,
        ax=ax3,
        pos=max_pos,
        )
    
    xgi.draw_hyperedges(max_H_sub, pos=max_pos, ax=ax3, alpha=0.1)

    cbar3 = fig.colorbar(
        node_col,
        ax=ax3,
        orientation="horizontal",
        pad=0.03,
        fraction=0.06,
    )
    cbar3.set_label(r"Initial $\theta$")
    cbar3.set_ticks([0, 2 * np.pi])
    cbar3.set_ticklabels([r"$0$", r"$2 \pi$"])
    cbar3.ax.yaxis.get_major_formatter().set_scientific(False)
    cbar3.ax.yaxis.get_major_formatter().set_useOffset(False)
    ax3.text(-0.05, 1.02, '(c)', transform=ax3.transAxes, fontweight='bold')
    ax3.text(0.60, 0.89, f'$N={N}, K={K}$', transform=ax3.transAxes, fontsize=8)

    ax3.set_axis_on()

    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = axes[1,1]

    max_init = np.array(MaxStateVectors.I)
    max_s1 = np.array(MaxStateVectors.S1)
    max_s2 = np.array(MaxStateVectors.S2)
    max_ex = max_s1 + max_s2
    max_d = np.array(MaxStateVectors.D)

    data = np.vstack([
        max_init % (2 * np.pi),
        max_s1 % (2 * np.pi),
        max_s2 % (2 * np.pi),
        max_ex % (2 * np.pi),
        max_d % (2 * np.pi)
        ])
    
    im = ax4.imshow(data, cmap="viridis", aspect="auto", interpolation="nearest", vmin=0, vmax=2*np.pi)
    
    cbar = fig.colorbar(im, ax=ax4, pad=0.02, fraction=0.046)
    cbar.ax.tick_params(labelsize=8, direction="in")
    cbar.set_label(r"Final $\theta$", fontsize=9)
    cbar.minorticks_on()
    
    row_labels = [r"$Initial$",r"$Single 1$", r"$Single 2$", r"$Expected$", r"$Double$"]
    ax4.set_yticks(np.arange(5))
    ax4.set_yticklabels(row_labels, fontsize=8)
    
    ax4.set_xticks(np.arange(0, 20, 2))  # Tick every 2 units for clean look
    ax4.set_xticks(np.arange(0, 20, 1), minor=True)
    ax4.set_xlabel("Node States", fontsize=9)
    
    ax4.text(-0.05, 1.02, '(d)', transform=ax4.transAxes, fontweight='bold')
    ax4.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)

    plt.tight_layout()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "extrema_kur_epis.png", format="png", dpi=300, bbox_inches="tight")


def plot_minimum_lo_epistasis_graphs(lower, seeds, N, p_L):
    idx = np.argmin(lower)
    min_seed = seeds[idx]

    G = nx.fast_gnp_random_graph(N, p_L, seed=int(min_seed))

    return 2


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


def main():
    set_figure_format()

    n_trials = 1000
    N = 20

    p_H = 0.03
    dp_L = p_H * (N - 2) * (1 / 2) 
    ep_L = p_H * (N - 2) * (1 / 3) 
    p_L = min(dp_L, 1.0)

    K = 100
    pert_str = np.pi / 2

    master_rng = np.random.default_rng()
    seeds = master_rng.bit_generator.seed_seq.spawn(n_trials)

    EpisArr = calculate_epistasis(n_trials=n_trials, N=N, p_H=p_H, p_L=p_L, K=K, pert_str=pert_str, seeds=seeds)

    higherL2 = EpisArr.HL2
    lowerL2 = EpisArr.LL2

    plot_epistasis(higher=higherL2,lower=lowerL2, N=N, p_H=p_H, p_L=p_L, K=K, pert_str=pert_str)

    plot_minimum_hi_epistasis_graphs(higher=higherL2,seeds=seeds, N=N, p_H=p_H, K=K, pert_str=pert_str)

    plt.show()


if __name__ == "__main__":
    main()

