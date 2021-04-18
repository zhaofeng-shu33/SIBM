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

def construct_h(data, p0, p1):
    '''p0, p1, vectors with length |\mathcal{X}|
    '''
    n, m = data.shape
    h = np.zeros([n])
    for i in range(n):
        for j in range(m):
            index = data[i, j]
            h[i] += np.log(p0[index] / p1[index])
    return h

def construct_B_tilde(B, data, p0, p1, a_b_ratio):
    h = construct_h(data, p0, p1)
    h /= np.log(a_b_ratio)
    n = B.shape[0]
    B_tilde = np.zeros([n + 1, n + 1])
    B_tilde[1:, 1:] = B / 2
    B_tilde[0, 1:] = h
    B_tilde[1:, 0] = h
    return B_tilde

def generate_data(ground_truth, n, m, p0, p1):
    # n, m = data.shape
    data = np.zeros([n, m], dtype=int)
    for i in range(n):
        if ground_truth[i] == 1:
            p = p0
        else:
            p = p1
        data[i, :] = np.random.choice([0, 1], size=m, p=p)
    return data