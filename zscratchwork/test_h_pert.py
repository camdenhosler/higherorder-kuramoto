#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xgi
from itertools import permutations

from src.perturbations import higher_perturbation, projection_distance
from src.higher_oscillators import build_sparse_A

def adjacency_tensor(H, order):
    N = H.num_nodes
    shape = tuple([N] * (order + 1))
    tensor = np.zeros(shape)

    edges = H.edges.filterby("order", order)
    for _, members in edges.members(dtype=dict).items():
        for idcs in permutations(members):
            tensor[idcs] = 1

    return tensor

def main():
    ps = [0.0,0.3]
    N = 100
    H = xgi.fast_random_hypergraph(N, ps, seed=2)
    A = adjacency_tensor(H,2)
    sparse_A = build_sparse_A(A)

    K = 100.0
    rng = np.random.default_rng()
    omega = rng.normal(loc=5.0, scale=1.0, size=N)

    t_start, t_end = 0.0, 30.0
    t_eval = np.linspace(t_start, t_end, 3000)

    theta_init = rng.uniform(low=0.0, high=2*np.pi, size=N)

    H._node_attr.update({
    node: {"angle": theta_init[node]} 
    for node in H.nodes
    })

    sim_params = {
        'omega': omega,
        'K': K,
        'N': N,
        'sparse_A': sparse_A
    }
    #for each epistasis might need a distribution especially if I continue working with random vectors
    node1 = rng.integers(0, 100)
    node2 = rng.integers(0, 100)
    fsfpa, ssfpa, dfpa = higher_perturbation(node1=node1,node2=node2,t_start=t_start,t_end=t_end,theta_init=theta_init,t_eval=t_eval,params=sim_params)
    print(projection_distance(fsfpa,ssfpa,dfpa))

if __name__ == "__main__":
    main()
