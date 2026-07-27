#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import xgi
import networkx as nx
import matplotlib.pyplot as plt

from src.sparsify import sparse_adjacency_matrix, sparse_adjacency_tensor
from collections import namedtuple
from juliacall import Main as jl

def generate_tensors(n_trials, N, p_H, p_L, master_rng: np.random.Generator | None = None):
    if master_rng is None:
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
    node_pool = np.arange(1, N)

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

def main(n_trials, N, p_H, p_L, K, pert_str):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(script_directory)

    jl.seval(f'import Pkg; Pkg.activate(raw"{parent_directory}")')


    script_dir = Path(__file__).parent
    julia_file_path = script_dir.parent / "src" / "perturbations.jl"

    jl.include(str(julia_file_path))
    
    batches = generate_tensors(n_trials,N,p_H,p_L)

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
    #LowerEpisData = jl.calc_l_epistasis(batches['theta_init'], batches['node1'], batches['node2'], H_params)

    plt.hist(HigherEpisData.nonrelTotalDist)
    plt.show()



if __name__ == "__main__":
    n_trials = 1000
    N = 20

    p_H = 0.03
    dp_L = p_H * (N - 2) * (1 / 2) 
    ep_L = p_H * (N - 2) * (1 / 3) 
    p_L = min(dp_L, 1.0)

    K = 100.0
    pert_str = np.pi / 2

    main(n_trials, N, p_H, p_L, K, pert_str)
