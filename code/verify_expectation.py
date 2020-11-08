import argparse
import numpy as np
from sbm import sbm_graph
def get_average(G, a, b, beta):
    n = len(G.nodes)
    L = np.zeros([n])
    for u, v in G.edges:
        if G.nodes[u]['block'] == G.nodes[v]['block']:
            L[u] -= 1
            L[v] -= 1
        else:
            L[u] += 1
            L[v] += 1
    val = np.sum(np.exp(beta * L))
    g = lambda x: (b * np.exp(x) + a * np.exp(-x)) / 2 - (a + b) / 2 + 1
    return val / np.power(n, g(beta))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=16.0)
    parser.add_argument('--b', type=float, default=4.0)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--beta', type=float, default=0.4, nargs='+')
    parser.add_argument('--repeat', type=int, default=1000, help='number of times to generate the SBM graph')
    args = parser.parse_args()
    val = 0
    for _ in range(args.repeat):
        G = sbm_graph(args.n, 2, args.a, args.b)
        val += get_average(G, args.a, args.b, args.beta)
    val /= args.repeat
    print(val)
