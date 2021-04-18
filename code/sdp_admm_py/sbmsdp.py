import networkx as nx
import numpy as np
    
def sbm2(n, a, b):
    if n % 2 != 0:
        raise ValueError('n % 2 != 0')
    elif a <= b:
        raise ValueError('a <= b')
    sizes = [int(n/2) for _ in range(2)]
    _p = np.log(n) * a / n
    _q = np.log(n) * b / n
    if _p > 1 or _q > 1:
        raise ValueError('%f (probability) larger than 1' % _p)
    p = np.diag(np.ones(2) * (_p - _q)) + _q * np.ones([2, 2])
    return nx.generators.community.stochastic_block_model(sizes, p)

def sdp2(G, kappa=1.0, rho = 0.1, max_iter = 1000, tol=1e-4):
    """Recover node labels of SBM with two communties by SDP.
    Parameters
    ----------
    rho : ADMM penalty parameter
    
    """
    B = _construct_B(G, kappa)
    n = len(G.nodes)
    X = np.zeros([n, n])
    U = np.zeros([n, n])
    Z = np.zeros([n, n])
    for _ in range(max_iter):
        X_new = Z - U + B / rho
        np.fill_diagonal(X_new, 1)
        X = X_new
        Z = _project_cone(X + U)
        delta_U = X - Z
        if np.linalg.norm(delta_U, ord='fro') < tol:
            break
        U = U + delta_U
    return _get_labels_sdp2(X)

def _project_cone(Y):
    vals, vectors = np.linalg.eigh(Y)
    vals = (vals + np.abs(-vals)) / 2
    return vectors @ np.diag(vals) @ vectors.T

def _construct_B(G, kappa=1):
    n = len(G.nodes)
    B = np.ones([n, n]) * (-kappa)
    for i, j in G.edges():
        B[i, j] = 1
        B[j, i] = 1
    return B

def _get_labels_sdp2(cluster_matrix):
    labels = cluster_matrix[0, :] < 0
    return labels.astype(np.int)