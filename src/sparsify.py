import numpy as np
import networkx as nx
from itertools import permutations

def sparse_adjacency_tensor(H, order):
    edges = H.edges.filterby("order", order)
    triplets = set()

    for _, members in edges.members(dtype=dict).items():
        for idcs in permutations(members):
            triplets.add(idcs)

    if not triplets:
        idx_arrays = tuple(np.array([], dtype=np.int64) for _ in range(order + 1))
        return idx_arrays + (np.array([], dtype=np.float64),)
    
    idx_arrays = tuple(np.array(coords, dtype=np.int64) for coords in zip(*triplets))
    vals = np.ones(len(triplets), dtype=np.float64)
    return idx_arrays + (vals,)

def sparse_adjacency_matrix(G):
    adj = nx.to_scipy_sparse_array(G, format="coo", dtype=np.float64)

    idx_i = np.concatenate([adj.row, adj.col])
    idx_j = np.concatenate([adj.col, adj.row])
    vals = np.ones(len(idx_i), dtype=np.float64)
    return idx_i, idx_j, vals