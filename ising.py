'''using Metropolis algorithm
to get a sample from Ising distribution
'''
import random

import numpy as np
from networkx import nx

def estimate_a_b(graph, k=2):
    '''for multiple communities
    '''
    n = len(graph.nodes)
    e = len(graph.edges)
    t = 0
    for _, v in nx.algorithms.cluster.triangles(graph).items():
        t += v
    t /= 3
    eq_1 = e / (n * np.log(n))
    eq_2 = t / (np.log(n) ** 3)
    # solve b first
    coeff_3 = -1 * (k - 1)
    coeff_2 = 6 * (k - 1) * eq_1
    coeff_1 = -12 * (k - 1) * (eq_1 ** 2)
    coeff_0 = 8 * k * (eq_1 ** 3) - 6 * eq_2
    coeff = [coeff_3, coeff_2, coeff_1, coeff_0]
    b = -1
    for r in np.roots(coeff):
        if abs(np.imag(r)) < 1e-10:
            b = np.real(r)
            break
    a = 2 * k * eq_1 - (k - 1) * b
    if b < 0 or b > a:
        raise ValueError('')
    return (a, b)

def _estimate_a_b(graph):
    '''for two communities
    '''
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
    def __init__(self, graph, k=2):
        self.G = graph
        self.k = k
        a, b = estimate_a_b(graph, k)
        _beta_star = np.log((a + b - k - np.sqrt((a + b - k)**2 - 4 * a * b))/ (2 * b))
        self._beta = 1.2 * _beta_star
        self._alpha = 1.8 * b * self._beta
        self.n = len(self.G.nodes)
        # randomly initiate a configuration
        self.sigma = [1 for i in range(self.n)]
        nodes = list(self.G)
        random.Random().shuffle(nodes)
        for node_state in range(k):
            for i in range(self.n // k):
                self.sigma[nodes[i * k + node_state]] = node_state
        self.mixed_param = self._alpha * np.log(self.n)
        self.mixed_param /= (self._beta * self.n)
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
    def metropolis(self, N=40):
        # iterate given rounds
        for _ in range(N):
            for r in range(self.n):
                # randomly select one new flipping state to inspect
                w_s = random.randint(1, self.k - 1)
                delta_H = self.get_dH(r, w_s)
                if delta_H < 0:  # lower energy: flip for sure
                    self.sigma[r] = (w_s + self.sigma[r]) % self.k
                else:  # Higher energy: flip sometimes
                    probability = np.exp(- self._beta * delta_H)
                    if np.random.rand() < probability:
                        self.sigma[r] = (w_s + self.sigma[r]) % self.k
        return self.sigma

def SIBM_metropolis(graph, k=2, max_iter=40):
    sibm = SIBM(graph, k)
    return sibm.metropolis(N=max_iter)