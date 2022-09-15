import numpy as np
from sdp import construct_B, project_cone
from sbm import sbm_graph, get_ground_truth
import matplotlib.pyplot as plt

def compare(label_1, label_2):
    n = len(label_1)
    invert_label_1 = np.ones(n) - label_1
    acc_1 = np.sum(np.array(label_1) == label_2) / n
    acc_2 = np.sum(np.array(invert_label_1) == label_2) / n
    if acc_1 > acc_2:
        return acc_1
    return acc_2

def get_labels_sdp2(cluster_matrix):
    w, v = np.linalg.eigh(cluster_matrix)
    labels =  v[:, -1] < 0
    return labels.astype(int)

def sdp2(G, kappa=1.0, rho = 0.1, max_iter = 100, tol=1e-4):
    '''only for two communities
    rho: ADMM penalty parameter
    '''
    B = construct_B(G, kappa)
    np.fill_diagonal(B, 0)
    gt = get_ground_truth(graph)
    n = len(G.nodes)
    u = np.zeros([n])
    Z = np.zeros([n, n])
    acc_list = np.zeros(max_iter)
    residue_list = np.zeros(max_iter)
    for i in range(max_iter):
        Z = project_cone(np.eye(n) + np.diag(u) + B / rho + Z - np.diag(np.diag(Z)))
        residue_list[i] = np.linalg.norm(np.ones([n]) - np.diag(Z))
        u = u + np.ones([n]) - np.diag(Z)
        labels = get_labels_sdp2(Z)
        acc_list[i] = compare(gt, labels)
    import pdb
    pdb.set_trace()
    return acc_list, residue_list

if __name__ == '__main__':
    n = 300
    k = 2
    a = 16
    b = 4
    graph = sbm_graph(n, k, a, b)
    acc_list, residue_list = sdp2(graph)
    plt.plot(acc_list)
    plt.figure(2)
    plt.plot(residue_list)
    plt.show()