'''using Metropolis algorithm
to get a sample from Potts distribution
'''
import random

import numpy as np
import networkx as nx

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


class SIBM:
    def _estimate_beta(self, repeat=40, acceptance_ratio=0.8):
        # Johnson's empirical method
        # - log(\chi_0) / Delta E 
        _energy = 0
        _average = 0
        for _ in range(repeat):
            r = np.random.randint(0, self.n)
            w_s = np.random.randint(1, self.k)
            delta_H = self.get_dH(r, w_s)
            if delta_H > 0:
                _energy += delta_H
                _average += 1
        _energy /= _average
        return -1.0 * _energy / np.log(acceptance_ratio)

    def __init__(self, graph, k=2, estimate_a_b_indicator=True, 
                 epsilon=0.0, _gamma=None, _beta=None):
        self.G = graph
        self.k = k
        self.n = len(self.G.nodes)
        self.epsilon = 0
        if estimate_a_b_indicator:
            try:
                a, b = estimate_a_b(graph, k)
            except ValueError:
                a, b = k, k
            square_term = (a + b - k)**2 - 4 * a * b
            if square_term > 0:
                _beta_star = np.log((a + b - k - np.sqrt(square_term))/ (2 * b))
                self._beta = 1.2 * _beta_star
                self._gamma = 13 * b
            else:
                self._beta = 1.2
                # the following choice of parameter is to guarantee gamma > b
                self._gamma = 2 * len(graph.edges) / (self.n * np.log(self.n))
                # (self._gamma, self._beta)
        else:
            self._beta = 1.2
            self._gamma = 13
        if _beta != None:
            self._beta = _beta
        # if _gamma != None:
        #    self._gamma = _gamma
        self.mixed_param = self._gamma * np.log(self.n)
        self.mixed_param /= self.n
        # randomly initiate a configuration
        self.sigma = [1 for i in range(self.n)]
        nodes = list(self.G)
        np.random.shuffle(nodes)
        for node_state in range(k):
            for i in range(self.n // k):
                self.sigma[nodes[i * k + node_state]] = node_state

        if self._beta < 0:
            self._beta = self._estimate_beta()

    
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
        r = np.random.randint(0, self.n)
        # randomly select one new flipping state to inspect
        w_s = np.random.randint(1, self.k)
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
            self._beta *= (1 + self.epsilon)
        return self.sigma

def potts_metropolis(graph, k=2, max_iter=40,
                     gamma=None, beta=None, seed=None):
    if seed:
        np.random.seed(seed)
    sibm = SIBM(graph, k, _gamma=gamma, _beta=beta)
    return sibm.metropolis(N=max_iter)