import numpy as np
import networkx as nx
from sdp_admm_py import sdp1_admm_py
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

sizes = [20, 20, 20]
_p = 0.9
_q = 0.4
k = 3
p = np.diag(np.ones(k) * (_p - _q)) + _q * np.ones([k, k])
g = nx.generators.community.stochastic_block_model(sizes, p)
a = nx.adjacency_matrix(g)
result = sdp1_admm_py(a, k)
print(get_labels(result))
