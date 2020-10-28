import random
import numpy as np

from sbm import sbm_graph
from sbm import compare, get_ground_truth
import argparse

class SIBM2:
    '''SIBM with two community'''
    def __init__(self, graph, _alpha, _beta):
        self.G = graph
        self._beta = _beta
        self._alpha_divide_beta = _alpha / self._beta
        self.n = len(self.G.nodes)
        # randomly initiate a configuration
        self.sigma = [1 for i in range(self.n)]
        nodes = list(self.G)
        random.Random().shuffle(nodes)
        k = 2
        self.k = k
        for node_state in range(k):
            for i in range(self.n // k):
                self.sigma[nodes[i * k + node_state]] = 2 * node_state - 1
        # node state is +1 or -1
        self.m = self.n / 2 # number of +1
        self.mixed_param = self._alpha_divide_beta * np.log(self.n)
        self.mixed_param /= self.n
    def get_dH(self, trial_location):
        _sum = 0
        w_s = self.sigma[trial_location]
        for i in self.G[trial_location]:
                _sum += self.sigma[i]
        _sum *= w_s * (1 + self.mixed_param)
        _sum -= self.mixed_param * (w_s * (2 * self.m - self.n)- 1)
        return _sum
    def _metropolis_single(self):
        # randomly select one position to inspect
        r = random.randint(0, self.n - 1)
        delta_H = self.get_dH(r)
        if delta_H < 0:  # lower energy: flip for sure
            self.sigma[r] *= -1
            self.m += self.sigma[r]
        else:  # Higher energy: flip sometimes
            probability = np.exp(-1.0 * self._beta * delta_H)
            if np.random.rand() < probability:
                self.sigma[r] *= -1
                self.m += self.sigma[r]
    def metropolis(self, N=40):
        # iterate given rounds
        for _ in range(N):
            for _ in range(self.n):
                self._metropolis_single()
        return self.sigma

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=16.0)
    parser.add_argument('--b', type=float, default=4.0)
    parser.add_argument('--n', type=int, default=300)
    # parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--repeat', type=int, default=1, help='number of times to generate the SBM graph')
    parser.add_argument('--inner_repeat', type=int, default=1000, help='number of sigma generated for a given graph')
    parser.add_argument('--max_iter', type=int, default=100)
    args = parser.parse_args()
    averaged_acc = 0
    k = 2
    for i in range(args.repeat):
        G = sbm_graph(args.n, k, args.a, args.b)    
        gt = get_ground_truth(G)
        sibm = SIBM2(G, args.alpha, args.beta)
        sibm.metropolis(N=args.max_iter) # burn-in period
        averaged_inner_acc = 0
        for i in range(args.inner_repeat):
            sibm._metropolis_single()
            inner_acc = compare(gt, sibm.sigma)
            inner_acc = int(inner_acc) # for exact recovery
            averaged_inner_acc += inner_acc
        averaged_inner_acc /= args.inner_repeat
        averaged_acc += averaged_inner_acc
    averaged_acc /= args.repeat
    print(averaged_acc)