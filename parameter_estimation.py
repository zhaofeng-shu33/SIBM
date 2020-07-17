'''
   parameter estimation validation for SSBM(n, k, p, q)
   with p = a log(n)/ n, q = b log(n) / n
'''
import argparse

from sbm import sbm_graph
from ising import estimate_a_b

def evaluation(n, k, a, b):
    graph = sbm_graph(n, k, a, b)
    _a, _b = estimate_a_b(graph, k)
    return (a - _a) **2 + (b - _b) ** 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--a', type=float, default=16)
    parser.add_argument('--b', type=float, default=4)
    parser.add_argument('--k', type=int, default=2)
    args = parser.parse_args()
    print(evaluation(args.n, args.k, args.a, args.b))