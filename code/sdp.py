import numpy as np
import networkx as nx
try:
    from sdp_admm_py import sdp1_admm_py
except ImportError:
    pass
from cvxopt import matrix, solvers
def solve_0_1(B):
    n = B.shape[0]
    h = matrix([[matrix(-B), matrix(0, (1, n))], [matrix(0, (n+1, 1))]])
    c  = matrix(0.0, (n + 1, 1))
    c[n] = 1
    g_list = []
    for i in range(n):
        matrix_temp = matrix(0.0, (n+1, n+1))
        matrix_temp[0, 0] = -1
        matrix_temp[n, i] = 0.5
        matrix_temp[i, n] = 0.5
        g_list.append(matrix_temp)
    matrix_last = matrix(0.0, (n+1, n+1))
    matrix_last[n, n] = -1
    g_list.append(matrix_last)
    sol = solvers.sdp(c, Gs=g_list, hs=h)
    return sol['zs']

def construct_B(G, kappa=1):
    n = len(G.nodes)
    B = np.ones([n, n]) * (-kappa)
    for i, j in G.edges():
        B[i, j] = 1
        B[j, i] = 1
    return B

def project_cone(Y):
    vals, vectors = np.linalg.eigh(Y)
    vals = (vals + np.abs(-vals)) / 2
    return vectors @ np.diag(vals) @ vectors.T

def sdp2(G, rho = 0.1, max_iter = 1000, tol=1e-4):
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
        np.fill_diagonal(X_new, 1)
        X = X_new
        Z = project_cone(X + U)
        delta_U = X - Z
        if np.linalg.norm(delta_U, ord='fro') < tol:
            break
        U = U + delta_U
    return get_labels_sdp2(X)

def get_labels_sdp2(cluster_matrix):
    labels = cluster_matrix[0, :] < 0
    return labels.astype(np.int)

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