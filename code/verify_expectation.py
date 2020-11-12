# verify \sum exp(beta * (B_i - A_i )) = n^{g(beta)}
import argparse
import logging

import numpy as np
from matplotlib import pyplot as plt

from sbm import sbm_graph

def theoretical(n, a, b):
    beta_star = np.log((a + b -2 - np.sqrt((a + b - 2) ** 2 - 4 * a * b)) / (2 * b))
    beta = np.linspace(0, beta_star)
    coeffient = 1 - np.log(n) ** 2 / (4 * n) * (a ** 2 * (np.exp(-1 * beta) - 1) ** 2 + b ** 2 * (np.exp(beta) - 1) ** 2)
    plt.plot(beta, coeffient)
    plt.show()

def get_average(G, a, b, beta, filter_positive=True):
    n = len(G.nodes)
    L = np.zeros([n])
    for u, v in G.edges:
        if G.nodes[u]['block'] == G.nodes[v]['block']:
            L[u] -= 1
            L[v] -= 1
        else:
            L[u] += 1
            L[v] += 1
    if filter_positive and np.any(L >= 0):
        return 0
    val = np.sum(np.exp(beta * L))
    g = lambda x: (b * np.exp(x) + a * np.exp(-x)) / 2 - (a + b) / 2 + 1
    return val / np.power(n, g(beta))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=16.0)
    parser.add_argument('--b', type=float, default=4.0)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--repeat', type=int, default=1500, help='number of times to generate the SBM graph')
    args = parser.parse_args()
    val = 0
    for _ in range(args.repeat):
        G = sbm_graph(args.n, 2, args.a, args.b)
        val += get_average(G, args.a, args.b, args.beta)
    val /= args.repeat
    print(val)
