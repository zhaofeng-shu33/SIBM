import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from potts import potts_metropolis

def convert_to_label_list(G, partition):
    g = list(G.nodes)
    n = len(g)
    cat = [0 for i in range(n)]
    label_index = 0
    for i in partition:
        for j in i:
            cat[g.index(j)] = label_index
        label_index += 1
    return cat

G = nx.read_gml('football.gml')
n = len(G.nodes)
cat = [G.nodes[i]['value'] for i in G.nodes]
k = max(cat) + 1

results_partition = nx.algorithms.community.modularity_max.greedy_modularity_communities(G)
cat_1 = convert_to_label_list(G, results_partition)
nmi = normalized_mutual_info_score(cat, cat_1)
print(nmi)

results_partition2 = nx.algorithms.community.asyn_fluidc(G, k)
cat_2 = convert_to_label_list(G, results_partition2)
nmi2 = normalized_mutual_info_score(cat, cat_2)
print(nmi2)

results_partition3 = nx.algorithms.community.louvain_communities(G)
cat_3 = convert_to_label_list(G, results_partition3)
nmi3 = normalized_mutual_info_score(cat, cat_3)
print(nmi3)

_G = nx.Graph()
g = list(G.nodes)
for edge in G.edges():
    _G.add_edge(g.index(edge[0]), g.index(edge[1]))
cat_4 = potts_metropolis(_G, k)
nmi4 = normalized_mutual_info_score(cat, cat_4)
print(nmi4)