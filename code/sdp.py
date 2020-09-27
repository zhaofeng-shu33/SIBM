import numpy as np
import networkx as nx
try:
    from sdp_admm_py import sdp1_admm_py
except ImportError:
    pass

def construct_B(G, kappa=-1):
    n = len(G.nodes)
    B = np.ones([n, n]) * (-kappa)
    for i, j in G.edges():
        B[i, j] = 1
    return B

def project_cone(Y):
    vals, vectors = np.linalg.eigh(Y)
    vals = (vals + np.abs(-vals)) / 2
    return vectors @ np.diag(vals) @ vectors.T

def sdp2(G, rho = 0.5, max_iter = 1000, tol=1e-4):
    '''only for two communties
    rho: ADMM penalty parameter
    '''
    B = construct_B(G)
    n = len(G.nodes)
    X = np.zeros([n, n])
    U = np.zeros([n, n])
    Z = np.zeros([n, n])
    for _ in range(max_iter):
        X_new = Z - U + B / rho
        if np.linalg.norm(X_new - X, ord='fro') < tol:
            break
        X = X_new
        np.fill_diagonal(X, 1)
        Z = project_cone(X + U)
        U += (X - Z)
    return get_labels(X)

def sdp(G, k):
    a = nx.adjacency_matrix(G)
    result = sdp1_admm_py(a, k, report_interval=10000)
    return get_labels(result)

def get_labels(cluster_matrix):
    n = cluster_matrix.shape[1]
    labels = np.zeros(n)
    label_index = 0
    for i in range(n):
       if labels[i] != 0:
            continue
       labels[i] = label_index
       for j in range(n):
           if cluster_matrix[i, j] > 0.5:
               labels[j] = label_index
       label_index = label_index + 1    
    return labels