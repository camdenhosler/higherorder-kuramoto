import numpy as np
import networkx as nx
import xgi
from itertools import permutations

from src.sparsify import sparse_adjacency_matrix, sparse_adjacency_tensor

def adjacency_tensor(H, order):
    N = H.num_nodes
    shape = tuple([N] * (order + 1))
    tensor = np.zeros(shape)

    edges = H.edges.filterby("order", order)
    for _, members in edges.members(dtype=dict).items():
        for idcs in permutations(members):
            tensor[idcs] = 1

    return tensor

def h_build_sparse_A(A):
    idx_i, idx_j, idx_k = np.nonzero(A)
    vals = np.empty(len(idx_i))
    for e in range(len(idx_i)):
        vals[e] = A[idx_i[e], idx_j[e], idx_k[e]]
    return idx_i, idx_j, idx_k, vals

def l_build_sparse_A(A):
    idx_i, idx_j = np.nonzero(A)
    vals = np.empty(len(idx_i))
    for e in range(len(idx_i)):
        vals[e] = A[idx_i[e], idx_j[e]]
    return idx_i, idx_j, vals

def test_higher_tensor_sparisfication():
    H = xgi.fast_random_hypergraph(50, [0,0.03])
    A_H = adjacency_tensor(H,2)
    o_idx_i, o_idx_j, o_idx_k, o_vals = h_build_sparse_A(A_H)
    n_idx_i, n_idx_j, n_idx_k, n_vals = sparse_adjacency_tensor(H,2)

    old_set = set(zip(o_idx_i, o_idx_j, o_idx_k))
    new_set = set(zip(n_idx_i, n_idx_j, n_idx_k))

    assert old_set == new_set, old_set.symmetric_difference(new_set)

def test_lower_tensor_sparisfication():
    G = nx.erdos_renyi_graph(50, 0.18)
    A_L = nx.to_numpy_array(G)
    o_idx_i, o_idx_j, o_vals = l_build_sparse_A(A_L)
    n_idx_i, n_idx_j, n_vals = sparse_adjacency_matrix(G)

    old_set = set(zip(o_idx_i, o_idx_j))
    new_set = set(zip(n_idx_i, n_idx_j))

    assert old_set == new_set, old_set.symmetric_difference(new_set)


