'''using Metropolis algorithm
to get a sample from Ising distribution
'''

import numpy as np
import random
class SIBM:
    def __init__(self, graph, a=16, b=4):
        self.G = graph
        _beta_star = 0.5 * np.log( (a + b - 2 - np.sqrt((a + b - 2)**2 - 4 * a * b))/ (2 * b))
        self._beta = 0.8 * _beta_star
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

def SIBM_metropolis(graph, a=16, b=4, max_iter=40):
    sibm = SIBM(graph, a=16, b=4)
    return sibm.metropolis(N=max_iter)