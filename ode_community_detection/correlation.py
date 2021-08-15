import numpy as np

def correlation_from_adj(adj_mat, delta_t):
    n, m = adj_mat.shape
    rho = np.zeros([n, n, m])
    for i in range(n):
        for j in range(n):
            whole = np.cos(adj_mat[i, :] - adj_mat[j, :])
            rho[i, j, 0] = whole[0]
            time = delta_t
            for r in range(1, m):
                rho[i, j, r] = (time * rho[i, j, r-1] + delta_t * np.sum(whole[r]))
                time += delta_t
                rho[i, j, r] /= time
    return rho
