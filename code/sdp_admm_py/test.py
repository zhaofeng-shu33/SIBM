import unittest

import numpy as np
import networkx as nx

from sdp_admm_py import sdp1_admm_py, sdp_admm_sbm_2_py
from utility import get_ground_truth, compare, construct_B
from sbmsdp import sbm2

def get_labels(cluster_matrix):
    n = cluster_matrix.shape[1]
    labels = np.zeros(n)
    label_index = 1
    for i in range(n):
       if labels[i] != 0:
            continue
       labels[i] = label_index
       for j in range(n):
           if cluster_matrix[i, j] > 0.5:
               labels[j] = label_index
       label_index = label_index + 1    
    return labels

class SDP_ADMM(unittest.TestCase):
    def test_basic(self):
        sizes = [20, 20, 20]
        _p = 0.9
        _q = 0.4
        k = 3
        p = np.diag(np.ones(k) * (_p - _q)) + _q * np.ones([k, k])
        g = nx.generators.community.stochastic_block_model(sizes, p)
        a = nx.adjacency_matrix(g)
        result = sdp1_admm_py(a, k)
        result_label = get_labels(result)
        self.assertAlmostEqual(np.sum(result_label[:20] == result_label[0]), 20)
        self.assertAlmostEqual(np.sum(result_label[20:40] == result_label[20]), 20)
        self.assertAlmostEqual(np.sum(result_label[40:] == result_label[40]), 20)
    def test_sdp2(self):
        G = sbm2(100, 16, 4)
        B = construct_B(G)
        results = sdp_admm_sbm_2_py(B, report_interval=10000)
        result_label = get_labels(results)
        labels_true = get_ground_truth(G)
        self.assertAlmostEqual(compare(result_label, labels_true), 1.0)

if __name__ == '__main__':
    unittest.main()