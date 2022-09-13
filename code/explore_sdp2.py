import numpy as np
from sdp import construct_B, project_cone, get_labels_sdp2
from sbm import sbm_graph, compare, get_ground_truth

def sdp2(G, kappa=1.0, rho = 0.1, max_iter = 1000, tol=1e-4):
    '''only for two communities
    rho: ADMM penalty parameter
    '''
    B = construct_B(G, kappa)

    n = len(G.nodes)
    u = np.zeros([n])
    Z = np.zeros([n, n])
    for i in range(max_iter):
        Z = project_cone(np.eye(n) + np.diag(u) + B / rho + Z - np.diag(np.diag(Z)))
        if np.linalg.norm(np.ones([n]) - np.diag(Z)) < tol:
            break
        u = u + np.ones([n]) - np.diag(Z)
    import pdb
    pdb.set_trace()
    return get_labels_sdp2(Z)

if __name__ == '__main__':
    n = 300
    k = 2
    a = 16
    b = 4
    graph = sbm_graph(n, k, a, b)
    labels = sdp2(graph)
    gt = get_ground_truth(graph)
    acc = compare(gt, labels)
    print(acc)