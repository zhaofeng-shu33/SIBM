import unittest

import numpy as np
import networkx as nx

class IsingModel:
    '''Ising model with two community on a general graph

    Parameters
    ----------
    graph: networkx like graph object
    _beta: inverse temperature
    '''
    def __init__(self, graph, _beta):
        self.G = graph
        self._beta = _beta
        self.n = len(self.G.nodes)
        # randomly initiate a configuration
        # node state is +1 or -1
        self.sigma = [2 * np.random.randint(0, 2) - 1 for i in range(self.n)]
    def get_dH(self, trial_location):
        _sum = 0
        w_s = self.sigma[trial_location]
        for i in self.G[trial_location]:
                _sum += self.sigma[i]
        _sum *= w_s
        return _sum
    def _metropolis_single(self):
        # randomly select one position to inspect
        r = np.random.randint(0, self.n)
        delta_H = self.get_dH(r)
        if delta_H < 0:  # lower energy: flip for sure
            self.sigma[r] *= -1
        else:  # Higher energy: flip sometimes
            probability = np.exp(-1.0 * self._beta * delta_H)
            if np.random.rand() < probability:
                self.sigma[r] *= -1
    def metropolis(self, N=40):
        # iterate given rounds
        for _ in range(N):
            for _ in range(self.n):
                self._metropolis_single()
        return self.sigma

# global parameters
beta = 0.1
P = np.array([
[1 - np.exp(-4 * beta), np.exp(-4 * beta) / 3, np.exp(-4 * beta) / 3, np.exp(-4 * beta) / 3, 0, 0, 0, 0],
[1 / 3.0, 0, 0, 0, 1 / 3.0, 1 / 3.0, 0, 0],
[1 / 3.0, 0, 0, 0, 1 / 3.0, 0, 1 / 3.0, 0],
[1 / 3.0, 0, 0, 0, 0, 1 / 3.0, 1 / 3.0, 0],
[0, 1 / 3.0, 1 / 3.0, 0, 0, 0, 0, 1 / 3.0],
[0, 1 / 3.0, 0, 1 / 3.0, 0, 0, 0, 1 / 3.0],
[0, 0, 1 / 3.0, 1 / 3.0, 0, 0, 0, 1 / 3.0],
[0, 0, 0, 0, np.exp(-4 * beta) / 3, np.exp(-4 * beta) / 3, np.exp(-4 * beta) / 3, 1 - np.exp(-4 * beta)]
])
p1 = np.exp(3 * beta) / (2 * np.exp(3 * beta) + 6 * np.exp(-1 * beta))
p2 = np.exp(-1 * beta) / (2 * np.exp(3 * beta) + 6 * np.exp(-1 * beta))

class Markov(unittest.TestCase):
    def test_eigen(self):
        values, vectors = np.linalg.eig(P.T)
        for i in range(8):
            if abs(values[i] - 1) < 1e-5:
                l2norm = np.linalg.norm(vectors[:, i], ord=1)
                _pi = vectors[:, i] / l2norm
                if _pi[0] < 0:
                    _pi = -1.0 * _pi
                self.assertAlmostEqual(_pi[0], p1)
                self.assertAlmostEqual(_pi[1], p2)
                self.assertAlmostEqual(_pi[2], p2)
                self.assertAlmostEqual(_pi[3], p2)
                self.assertAlmostEqual(_pi[4], p2)
                self.assertAlmostEqual(_pi[5], p2)
                self.assertAlmostEqual(_pi[6], p2)
                self.assertAlmostEqual(_pi[7], p1)
    def test_metropolis_sampling(self):
        G = nx.complete_graph(3)
        ising = IsingModel(G, beta)
        ising.metropolis()
        _pi = np.zeros([8])
        state_vectors = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                         [1,-1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
        iteration_time = 100000
        # np.random.seed(0)
        for _ in range(iteration_time):
            ising._metropolis_single()
            _pi[state_vectors.index(ising.sigma)] += 1
        _pi /= iteration_time
        print(_pi)
        print(p1, p2)
        print(np.sum(_pi))
        print(p1 * 2 + 6 * p2)
if __name__ == '__main__':
    unittest.main()