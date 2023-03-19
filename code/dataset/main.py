import argparse
import networkx as nx
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from potts import potts_metropolis
potts_parameters = {
    'karate': {'beta': 1.8, 'seed': 118}, 
    'football': {'beta': 2.4, 'seed': 118},
    'dolphins': {'beta': 2.7, 'seed': 118},
    'polbooks': {'beta': 1.8, 'seed': 118}
}
def convert_to_label_list(n, partition):
    cat = [0 for i in range(n)]
    label_index = 0
    for i in partition:
        for j in i:
            cat[j] = label_index
        label_index += 1
    return cat
if __name__ == '__main__':
    dataset_choices = ['karate', 'football', 'dolphins', 'polbooks']
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=dataset_choices, default='karate')
    args = parser.parse_args()
    dataset = args.dataset
    G = nx.read_gml(dataset + '.gml', label='id')
    n = len(G.nodes)
    cat = [G.nodes[i]['value'] for i in G.nodes]
    k = max(cat) + 1
    print(k)
    np.random.seed(118)
    results_partition = nx.algorithms.community.modularity_max.greedy_modularity_communities(G)
    cat_1 = convert_to_label_list(n, results_partition)
    nmi = normalized_mutual_info_score(cat, cat_1)
    print(nmi)

    results_partition2 = nx.algorithms.community.asyn_fluidc(G, k)
    cat_2 = convert_to_label_list(n, results_partition2)
    nmi2 = normalized_mutual_info_score(cat, cat_2)
    print(nmi2)

    results_partition3 = nx.algorithms.community.louvain_communities(G)
    cat_3 = convert_to_label_list(n, results_partition3)
    nmi3 = normalized_mutual_info_score(cat, cat_3)
    print(nmi3)



    #for i in range(100, 200):
    #for beta in np.arange(0.1, 3.8, 0.1):
    cat_4 = potts_metropolis(G, k, beta=potts_parameters[dataset]['beta'],
                            seed=potts_parameters[dataset]['seed'])
    nmi4 = normalized_mutual_info_score(cat, cat_4)
    print(nmi4)
    #    print(beta, nmi4)

    results_partition5 = nx.algorithms.community.label_propagation_communities(G)
    cat_5 = convert_to_label_list(n, results_partition5)
    nmi5 = normalized_mutual_info_score(cat, cat_5)
    print(nmi5)
