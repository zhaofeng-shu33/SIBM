import unittest

import numpy as np
import networkx as nx

from sbm import sbm_graph
from ising import SIBM
from sdp import sdp2, solve_sdp_cvx
from sbm import get_ground_truth, compare
from sibm_experiment import majority_voting, exact_compare_k
from sklearn.utils._testing import assert_array_equal

class TestSIBMExperiment(unittest.TestCase):
    def test_majority_voting(self):
        a = [[1, -1, 1, -1], [1, 1, 1, -1], [1, -1, 1, -1]]
        assert_array_equal(majority_voting(a), np.array([1, -1, 1, -1], dtype=int))
        a = [[1, -1, 1, -1], [1, 1, 1, -1], [1, -1, 1, 1]]
        assert_array_equal(majority_voting(a), np.array([1, -1, 1, -1], dtype=int))
        a = [[1, -1, 1, -1]]
        assert_array_equal(majority_voting(a), np.array([1, -1, 1, -1], dtype=int))
    def test_exact_compare_k(self):
        a = [0, 0, 0, 1, 1, 1]
        self.assertTrue(exact_compare_k(a, 2))
        a = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        self.assertTrue(exact_compare_k(a, 3))
        a = [0, 0, 0, 2, 2, 2, 1, 1, 1]
        self.assertTrue(exact_compare_k(a, 3))

class TestIsing(unittest.TestCase):
    def test_ising(self):
        # demo illustration to use sibm method for general graph
        G = sample_graph()
        sibm = SIBM(G, k=3, estimate_a_b_indicator=False, epsilon=0.1)
        print(sibm.metropolis())

class TestSDP(unittest.TestCase):
    def test_sdp2(self):
        G = sbm_graph(100, 2, 16, 4)
        results = sdp2(G)
        print(results)
        labels_true = get_ground_truth(G)
        self.assertAlmostEqual(compare(results, labels_true), 1.0)        
    def test_sdp2_svx(self):
        G = sbm_graph(100, 2, 16, 4)
        results = solve_sdp_cvx(G)
        labels_true = get_ground_truth(G)
        self.assertAlmostEqual(compare(results, labels_true), 1.0)        

def sample_graph():
    # see Network_Community_Structure.svg for an illustration
    # of this graph
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(4, 5)
    G.add_edge(4, 6)
    G.add_edge(4, 7)
    G.add_edge(4, 8)
    G.add_edge(5, 7)
    G.add_edge(6, 7)
    G.add_edge(6, 8)
    G.add_edge(9, 10)
    G.add_edge(9, 11)
    G.add_edge(9, 12)
    G.add_edge(9, 13)
    G.add_edge(10, 11)
    G.add_edge(10, 13)
    G.add_edge(11, 12)
    G.add_edge(12, 13)
    G.add_edge(1, 5)
    G.add_edge(1, 9)
    G.add_edge(6, 10)
    G.add_edge(7, 11)
    return G

if __name__ == '__main__':
    unittest.main()