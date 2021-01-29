import argparse
import numpy as np

from sbm import sbm_graph, get_ground_truth, compare
from sdp import sdp2_si

def generate_data(ground_truth, n, m, p0, p1):
    # n, m = data.shape
    data = np.zeros([n, m], dtype=int)
    for i in range(n):
        if ground_truth[i] == 1:
            p = p0
        else:
            p = p1
        data[i, :] = np.random.choice([0, 1], size=m, p=p)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=1, help='number of times to generate the SBM graph')
    parser.add_argument('--multi_thread', type=int, default=1)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--a', type=float, default=16.0)
    parser.add_argument('--b', type=float, default=4.0)
    parser.add_argument('--p0', type=float, default=[0.8, 0.2], nargs='+')
    parser.add_argument('--p1', type=float, default=[0.2, 0.8], nargs='+')
    args = parser.parse_args()
    acc = 0
    for _ in range(args.repeat):
        graph = sbm_graph(args.n, 2, args.a, args.b)
        gt = get_ground_truth(graph)
        a_b_ratio = args.a / args.b
        data = generate_data(gt, args.n, args.m, args.p0, args.p1)
        result_label = sdp2_si(graph, data, args.p0, args.p1, a_b_ratio, rho = 0.1, max_iter = 1000, tol=1e-4)
        acc += compare(gt, result_label)
    acc /= args.repeat
    print(acc)