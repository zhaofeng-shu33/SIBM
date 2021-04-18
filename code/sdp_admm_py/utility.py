import numpy as np
from sklearn import metrics

def get_ground_truth(graph):
    label_list = []
    for n in graph.nodes(data=True):
        label_list.append(n[1]['block'])
    return label_list

def compare(label_0, label_1):
    '''
    get acc using adjusted rand index
    '''
    return metrics.adjusted_rand_score(label_0, label_1)

def construct_B(G, kappa=1):
    n = len(G.nodes)
    B = np.ones([n, n]) * (-kappa)
    for i, j in G.edges():
        B[i, j] = 1
        B[j, i] = 1
    return B