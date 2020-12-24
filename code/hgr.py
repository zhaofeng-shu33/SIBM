import numpy as np
import networkx as nx

from sbm import sbm_graph

def hgr(G, p, q, N=100):
    n = len(G.nodes)
    hy = np.zeros([n])
    # Monte-Carlo simulation
    y_matrix = 2 * np.random.randint(0, 2, size=(n, N)) - 1
    # modify y_1 to 1
    y_matrix[0, :] = 1
    A = nx.adjacency_matrix(G).toarray()
    A_prime = np.ones([n, n]) - np.eye(n) - A
    for i in range(N):
        y = y_matrix[:, i]
        s2 = (1 - p) / (1 - q)
        hy[i] = np.power(p / q, y @ A @ y / 4) * np.power(s2, y @ A_prime @ y / 4)
    symbols = np.ones([n])
    for i in range(1, n):
        symbols[i] = np.average(hy[y_matrix[i, :] == 1]) - np.average(hy[y_matrix[i, :] == -1])
    return np.asarray(symbols > 0, dtype=int)

if __name__ == '__main__':
    n = 100
    a = 16
    b = 4
    G = sbm_graph(n, 2, a, b)
    p = a * np.log(n) / n
    q = b * np.log(n) / n
    labels = hgr(G, p, q)
    print(labels)