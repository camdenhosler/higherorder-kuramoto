#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

from src.perturbations import perturbation, projection_distance

def main():
    G = nx.watts_strogatz_graph(n=100, k=4, p=0.2, seed=1)
    A = nx.to_numpy_array(G)
    N = len(G)

    K = 100.0
    rng = np.random.default_rng()
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
    #for each epistasis might need a distribution especially if I continue working with random vectors
    node1 = rng.integers(0, 100)
    node2 = rng.integers(0, 100)
    fsfpa, ssfpa, dfpa = perturbation(node1=node1,node2=node2,t_start=t_start,t_end=t_end,theta_init=theta_init,t_eval=t_eval,params=sim_params)
    print(projection_distance(fsfpa,ssfpa,dfpa))

if __name__ == "__main__":
    main()
