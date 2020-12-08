import argparse

import numpy as np

from sbm import sbm_graph, get_ground_truth
from ising import SIBM_metropolis

def random_guess(graph):
    n = len(graph.nodes)
    return 2 * np.random.randint(0, 2, size=n) - 1

def metropolis_wrapper(graph):
    labels = SIBM_metropolis(graph)
    return 2 * np.array(labels) - 1

def correlation(ground_truth, estimated_label):
    return np.abs(np.dot(ground_truth, estimated_label))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', choices=['random', 'metropolis', 'sdp2'], default='metropolis')
    parser.add_argument('--repeat', type=int, default=100, help='number of times to generate the SBM graph')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--a', type=float, default=16.0, nargs='+')
    parser.add_argument('--b', type=float, default=4.0, nargs='+')
    args = parser.parse_args()

    n = args.n
    k = args.k
    a = args.a
    b = args.b
    repeat_time = args.repeat

    if args.alg == 'random':
        alg = random_guess
    elif args.alg == 'metropolis':
        alg = metropolis_wrapper

    corr_value = 0
    for i in range(repeat_time):
        graph = sbm_graph(n, k, a, b)
        gt = 2 * np.array(get_ground_truth(graph)) - 1
        corr_value_single = correlation(gt, alg(graph))
        corr_value += corr_value_single
    corr_value /= repeat_time
    print(corr_value)

