'''using Metropolis algorithm
to get a sample from Ising distribution
'''
import random

import numpy as np
from networkx import nx

def estimate_a_b(graph):
    n = len(graph.nodes)
    e = len(graph.edges)
    t = 0
    for _, v in nx.algorithms.cluster.triangles(graph).items():
        t += v
    t /= 3
    eq_1 = 4 * e / (n * np.log(n))
    eq_2 = 8 * t / (np.log(n) ** 3)
    # solve a first
    coeff = [4 / 3.0, -2 * eq_1, eq_1 ** 2, - eq_2]
    a = -1
    for r in np.roots(coeff):
        if abs(np.imag(r)) < 1e-10:
            a = np.real(r)
            break
    if a > eq_1 or a < eq_1 / 2:
        raise ValueError('')
    b = eq_1 - a
    return (a, b)

class SIBM:
    def __init__(self, graph):
        self.G = graph
        a, b = estimate_a_b(graph)
        _beta_star = 0.5 * np.log( (a + b - 2 - np.sqrt((a + b - 2)**2 - 4 * a * b))/ (2 * b))
        self._beta = 1.2 * _beta_star
        self._alpha = 1.8 * b * self._beta
        self.n = len(self.G.nodes)
        # randomly initiate a configuration
        self.sigma = [1 for i in range(self.n)]
        nodes = list(self.G)
        random.Random().shuffle(nodes)
        for i in range(self.n // 2):
            self.sigma[nodes[i]] = -1
        self.mixed_param = 2 * self._alpha * np.log(self.n)
        self.mixed_param /= (self._beta * self.n)
    def get_dH(self, trial_location):
        _sum = 0
        for i in range(self.n):
            if self.G.has_edge(trial_location, i):
                _sum += 2 * self.sigma[trial_location] * self.sigma[i]
            else:
                _sum -= self.mixed_param * self.sigma[trial_location] * self.sigma[i]
        return _sum
    def metropolis(self, N=40):
        # iterate given rounds
        for _ in range(N):
            # randomly select one vertex to inspect
            for r in range(self.n):
                delta_H = self.get_dH(r)
                if delta_H < 0:  # lower energy: flip for sure
                    self.sigma[r] = -self.sigma[r]
                else:  # Higher energy: flip sometimes
                    probability = np.exp(- self._beta * delta_H)
                    if np.random.rand() < probability:
                        self.sigma[r] = -self.sigma[r]
        s0 = set([i for i in range(self.n) if self.sigma[i] == 1])
        s1 = set([i for i in range(self.n) if self.sigma[i] == -1])
        return (s0, s1)

def SIBM_metropolis(graph, max_iter=40):
    sibm = SIBM(graph)
    return sibm.metropolis(N=max_iter)