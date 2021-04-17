import os
import numpy as np
import networkx as nx
try:
    from sdp_admm_py import sdp1_admm_py
except ImportError:
    pass
try:
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
except ImportError:
    pass

def solve_sdp_cvx(G):
    B = construct_B(G)
    n = B.shape[0]
    h = [matrix(-B)]
    c  = matrix(1.0, (n, 1))
    g_matrix = matrix(0.0, (n * n, n))
    for i in range(n):
        g_matrix[n * i + i, i] = -1
    sol = solvers.sdp(c, Gs=[g_matrix], hs=h)
    return get_labels_sdp2(np.array(sol['zs'][0]))

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

def construct_h(data, p0, p1):
    '''p0, p1, vectors with length |\mathcal{X}|
    '''
    n, m = data.shape
    h = np.zeros([n])
    for i in range(n):
        for j in range(m):
            index = data[i, j]
            h[i] += np.log(p0[index] / p1[index])
    return h

def construct_B_tilde(B, data, p0, p1, a_b_ratio):
    h = construct_h(data, p0, p1)
    h /= np.log(a_b_ratio)
    n = B.shape[0]
    B_tilde = np.zeros([n + 1, n + 1])
    B_tilde[1:, 1:] = B / 2
    B_tilde[0, 1:] = h
    B_tilde[1:, 0] = h
    return B_tilde

def admm_inner(B, b, rho = 0.1, max_iter = 1000, tol=1e-4):
    '''b: 2n vector
    '''
    n = B.shape[0]
    X = np.zeros([n, n])
    U = np.zeros([n, n])
    Z = np.zeros([n, n])
    for _ in range(max_iter):
        X_new = Z - U + B / rho
        X = project_A(X_new, b, n)
        Z = project_cone(X + U)
        delta_U = X - Z
        if np.linalg.norm(delta_U, ord='fro') < tol:
            break
        U = U + delta_U
    return X

def project_A(X0, b, n):
    return X0 - Acs(Pinv(Ac(X0, n) - b, n), n)


def Acs(z, n):
    mu = z[:n]
    nu = z[n:]
    Z = np.zeros([n, n])
    np.fill_diagonal(Z, nu)
  
    for i in range(1, n):
        for j in range(i + 1, n):
            Z[i, j] = mu[i] + mu[j]
            Z[j, i] = Z[i, j]
    Z[0, 1:] = mu[0]
    Z[1:, 0] = mu[0]
    return Z

def Pinv(z, n):
    mu = z[1:n]
    nu = z[n:]
    return np.concatenate(([z[0] / (2 * n - 2)], (1. / (2 * (n - 3))) * (mu - np.ones(n - 1) * np.sum(mu)/ (2 * n - 4)), nu), axis=None)


def Ac(X, n):
    return np.concatenate(([2 * np.sum(X[0, 1:])], 2 * (X[1:, 1:] - np.diag(np.diag(X[1:, 1:]))) @ np.ones([n - 1]), np.diag(X)), axis=None)


def sdp2_si(G, data, p0, p1, a_b_ratio, rho = 0.1, max_iter = 1000, tol=1e-4):
    '''only for two communties, with side information
       data: n times m n-array
       rho: ADMM penalty parameter
    '''
    B = construct_B(G, 1.0)
    B_tilde = construct_B_tilde(B, data, p0, p1, a_b_ratio)
    n = len(G.nodes)
    b = np.zeros([2 * n + 2])
    b[:n + 1] = -2
    b[0] = 0
    b[n + 1:] = 1
    try:
        from sdp_admm_py import sdp1_admm_si_py
        X = sdp1_admm_si_py(B_tilde, rho, max_iter, tol, report_interval=20000)
    except ImportError:
        X = admm_inner(B_tilde, b, rho, max_iter, tol)
    labels = X[0, 1:] > 0
    return labels.astype(np.int)
 
def sdp2(G, kappa=1.0, rho = 0.1, max_iter = 1000, tol=1e-4):
    '''only for two communties
    rho: ADMM penalty parameter
    '''
    B = construct_B(G, kappa)
    if os.environ.get('DISABLE_EIGEN') is None:
        from sdp_admm_py import sdp_admm_sbm_2_py
        X = sdp_admm_sbm_2_py(B, rho, max_iter, tol, report_interval=20000)
    else:
        n = len(G.nodes)
        X = np.zeros([n, n])
        U = np.zeros([n, n])
        Z = np.zeros([n, n])
        for _ in range(max_iter):
            X = Z - U + B / rho
            np.fill_diagonal(X, 1)
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

