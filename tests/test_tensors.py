import numpy as np
import networkx as nx
import xgi
from itertools import permutations

import src.higher_oscillators as hi
import src.lower_oscillators as lo

def adjacency_tensor(H, order):
    N = H.num_nodes
    shape = tuple([N] * (order + 1))
    tensor = np.zeros(shape)

    edges = H.edges.filterby("order", order)
    for _, members in edges.members(dtype=dict).items():
        for idcs in permutations(members):
            tensor[idcs] = 1

    return tensor

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

def test_higher_tensor_sparisfication():
    H = xgi.fast_random_hypergraph(50, [0,0.03])
    A_H = adjacency_tensor(H,2)
    o_idx_i, o_idx_j, o_idx_k, o_vals = hi.build_sparse_A(A_H)
    n_idx_i, n_idx_j, n_idx_k, n_vals = sparse_adjacency_tensor(H,2)

    old_set = set(zip(o_idx_i, o_idx_j, o_idx_k))
    new_set = set(zip(n_idx_i, n_idx_j, n_idx_k))

    assert old_set == new_set, old_set.symmetric_difference(new_set)

def test_lower_tensor_sparisfication():
    G = nx.erdos_renyi_graph(50, 0.18)
    A_L = nx.to_numpy_array(G)
    o_idx_i, o_idx_j, o_vals = lo.build_sparse_A(A_L)
    n_idx_i, n_idx_j, n_vals = sparse_adjacency_matrix(G)

    old_set = set(zip(o_idx_i, o_idx_j))
    new_set = set(zip(n_idx_i, n_idx_j))

    assert old_set == new_set, old_set.symmetric_difference(new_set)


