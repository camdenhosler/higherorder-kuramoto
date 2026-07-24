#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import xgi
from itertools import permutations
from concurrent.futures import ProcessPoolExecutor

from src.higher_oscillators import build_sparse_A
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

def LO_run_trial(seed,N,K,t_start,t_end, t_eval, p):
    G = nx.erdos_renyi_graph(N, p, seed=int(seed))
    A = nx.to_numpy_array(G)

    rng = np.random.default_rng(seed)
    omega = rng.normal(loc=5.0, scale=1.0, size=N)
    theta_init = rng.uniform(low=0.0, high=2*np.pi, size=N)
    sim_params = {
        'omega': omega,
        'K': K,
        'N': N,
        'A': A
    }

    node1, node2 = rng.choice(N, size=2, replace=False)
    fsfpa, ssfpa, dfpa = perturbation(node1=node1,node2=node2,t_start=t_start,t_end=t_end,theta_init=theta_init,t_eval=t_eval, params=sim_params)

    return projection_distance(fsfpa,ssfpa,dfpa), seed

def HO_run_trial(seed,N,K,t_start,t_end,ps):
    H = xgi.fast_random_hypergraph(N, ps, seed=int(seed))
    A = adjacency_tensor(H,2)
    sparse_A = build_sparse_A(A)

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
    fsfpa, ssfpa, dfpa = perturbation(node1=node1,node2=node2,t_start=t_start,t_end=t_end,theta_init=theta_init,params=sim_params)

    return projection_distance(fsfpa,ssfpa,dfpa), seed

def LO_min_epistasis(num_trials, N):
    p_H = 0.004
    ep_L = p_H * (N - 2) * (1 / 3)
    p = ep_L

    t_start, t_end = 0.0, 30.0
    t_eval = np.linspace(t_start, t_end, 3000)
    K = 10.0
    master_rng = np.random.default_rng()
    seeds = master_rng.integers(0, 2**32 - 1, size=num_trials)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(LO_run_trial, seed, N, K, t_start, t_end, t_eval, p)
            for seed in seeds
        ]

        epis_arr_list = []
        for i, future in enumerate(futures):
            epis_arr_list.append(future.result())
            print(f"Trial {i+1}/{num_trials} complete")
        
    epis_arr = np.array(epis_arr_list)

    idx = np.argmin(epis_arr[:,0])
    min_epistasis = epis_arr[idx,0]
    min_seed = epis_arr[idx,1]

    G = nx.erdos_renyi_graph(N, p, seed=int(min_seed))

    min_rng = np.random.default_rng(int(min_seed))
    omega = min_rng.normal(loc=5.0, scale=1.0, size=N)
    theta_init = min_rng.uniform(low=0.0, high=2*np.pi, size=N)

    node1, node2 = min_rng.choice(N, size=2, replace=False)

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos, edge_color="slateblue", alpha=0.5)

    color = np.zeros(N)
    color[node1] = 1
    color[node2] = 1

    #print(theta_init)
    #print(omega)

    nx.draw_networkx_nodes(
    G, 
    pos,
    node_color=color, 
    cmap=plt.cm.RdBu, 
    vmin = -1,
    vmax = 1,
    node_size=50,
    edgecolors="black",
    linewidths=1,
    )
    plt.title(f"{min_epistasis}")
    plt.show()

    return epis_arr

def HIGH_min_epistasis(num_trials, N):
    p_H = 0.004
    ps = [0.0,p_H]
    K = 100.0
    t_start, t_end = 0.0, 25.0

    master_rng = np.random.default_rng()
    seeds = master_rng.integers(0, 2**32 - 1, size=num_trials)
    print(seeds)
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(HO_run_trial, seed, N, K, t_start, t_end, ps)
            for seed in seeds
        ]

        epis_arr_list = []
        for i, future in enumerate(futures):
            epis_arr_list.append(future.result())
            print(f"Trial {i+1}/{num_trials} complete")

    epis_arr = np.array(epis_arr_list)

    idx = np.argmin(epis_arr[:,0])
    min_epistasis = epis_arr[idx,0]
    min_seed = int(epis_arr[idx,1])

    H = xgi.fast_random_hypergraph(N, ps, seed=min_seed)

    rng = np.random.default_rng(min_seed)

    node1, node2 = rng.choice(N, size=2, replace=False)
    #omega = rng.normal(loc=5.0, scale=1.0, size=N)
    #theta_init = rng.uniform(low=0.0, high=2*np.pi, size=N)

    color = np.zeros(N)
    color[node1] = 1
    color[node2] = 1

    #print(theta_init)
    #print(omega)
    plt.figure(figsize=(8, 6))
    xgi.draw(H,
             node_fc=color,
             node_fc_cmap="RdBu",
             vmin = -1,
             vmax = 1,
             hull = True,
             )
    plt.title(f"{min_epistasis}")
    plt.show()


if __name__ == "__main__":
    N = 30
    num_trials = 100
    #LO_min_epistasis(num_trials, N)
    HIGH_min_epistasis(num_trials, N)
