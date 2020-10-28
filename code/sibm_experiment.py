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
                self.sigma[nodes[i * k + node_state]] = node_state
        self.m = self.n / 2
        self.mixed_param = self._alpha_divide_beta * np.log(self.n)
        self.mixed_param /= self.n
    def _get_Js(self, sigma_i, sigma_j, w_s):
        if sigma_i == sigma_j:
            return 1
        elif (w_s + sigma_i) % self.k == sigma_j:
            return -1
        else:
            return 0
    def get_dH(self, trial_location, w_s):
        _sum = 0
        for i in range(self.n):
            if self.G.has_edge(trial_location, i):
                _sum += self._get_Js(self.sigma[trial_location], self.sigma[i], w_s)
            elif trial_location != i:
                _sum -= self.mixed_param * self._get_Js(self.sigma[trial_location], self.sigma[i], w_s)
        return _sum
    def _get_Hamiltonian(self):
        H_value = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.sigma[i] != self.sigma[j]:
                    continue
                if self.G.has_edge(i, j):
                    H_value -= 1
                else:
                    H_value += self.mixed_param
        return H_value
    def _metropolis_single(self):
        # randomly select one position to inspect
        r = random.randint(0, self.n - 1)
        # randomly select one new flipping state to inspect
        w_s = random.randint(1, self.k - 1)
        delta_H = self.get_dH(r, w_s)
        if delta_H < 0:  # lower energy: flip for sure
            self.sigma[r] = (w_s + self.sigma[r]) % self.k
        else:  # Higher energy: flip sometimes
            probability = np.exp(-1.0 * self._beta * delta_H)
            if np.random.rand() < probability:
                self.sigma[r] = (w_s + self.sigma[r]) % self.k

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
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--repeat', type=int, default=1, help='number of times to generate the SBM graph')
    parser.add_argument('--inner_repeat', type=int, default=1000, help='number of sigma generated for a given graph')
    parser.add_argument('--max_iter', type=int, default=100)
    args = parser.parse_args()
    averaged_acc = 0
    for i in range(args.repeat):
        G = sbm_graph(args.n, args.k, args.a, args.b)    
        gt = get_ground_truth(G)
        sibm = SIBM(G, args.k, estimate_a_b_indicator=False,
                    _alpha=args.alpha, _beta=args.beta)
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