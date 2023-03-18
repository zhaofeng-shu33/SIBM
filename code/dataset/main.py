import argparse
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from potts import potts_metropolis

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

    cat_4 = potts_metropolis(G, k)
    nmi4 = normalized_mutual_info_score(cat, cat_4)
    print(nmi4)