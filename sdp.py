import numpy as np
import networkx as nx
try:
    from sdp_admm_py import sdp1_admm_py
except ImportError:
    pass
def sdp(G, k):
    a = nx.adjacency_matrix(G)
    result = sdp1_admm_py(a, k, report_interval=10000)
    return get_labels(result)

def get_labels(cluster_matrix):
    n = cluster_matrix.shape[1]
    labels = np.zeros(n)
    label_index = 0
    for i in range(n):
       if labels[i] != 0:
            continue
       labels[i] = label_index
       for j in range(n):
           if cluster_matrix[i, j] > 0.5:
               labels[j] = label_index
       label_index = label_index + 1    
    return labels