import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import cluster
from sklearn.metrics.cluster import normalized_mutual_info_score

from spectral_clustering import SpectralClustering
from sdp2_si_B import sdp2_si_B, sdp2

attr = pd.read_csv('LazegaLawyers/ELattr.dat', sep=' ', header=None)

ground_truth = attr[1].to_numpy() # choose status as ground truth

gender = attr[2]
office = pd.get_dummies(attr[3], prefix='office')
practice = attr[6]
law_school = pd.get_dummies(attr[7], prefix='law_school')

# one hot encoding
encoded = office.join(gender).join(practice).join(law_school).join(attr[4]).join(attr[5])
np_data = encoded.to_numpy()



# obtain spectral embedding 1D feature
sc = SpectralClustering(2, gamma=0.01)
sc.fit(np_data)
scaler = MinMaxScaler()
scaler.fit(sc.embedding_features[:,1].reshape(-1, 1))
sc_embedding_features= scaler.transform(sc.embedding_features[:,1].reshape(-1, 1))
sc_embedding_features = sc_embedding_features[:, 0]


# obtain KMeans clustering labels
kmeans_label = cluster.KMeans(n_clusters=2).fit(np_data).labels_


friends_matrix = pd.read_csv('LazegaLawyers/ELfriend.dat', sep=' ', header=None)
adj_matrix = friends_matrix.to_numpy()


# only use 70 records, 35 are lawyers with status "partner", 35 are with status "associate"
removed_index = 32
adj_matrix_new = np.delete(adj_matrix, removed_index, 0)
adj_matrix_new = np.delete(adj_matrix_new, removed_index, 1)
ground_truth_new = np.delete(ground_truth, removed_index, 0)
spectral_embedding_side_info = np.delete(sc_embedding_features, removed_index, 0)
kmeans_side_info = np.delete(kmeans_label, removed_index, 0)





G = nx.from_numpy_matrix(adj_matrix_new)
labels_sdp = sdp2(G)
nmi_spectral = normalized_mutual_info_score(ground_truth_new, spectral_embedding_side_info > np.median(spectral_embedding_side_info))
nmi_sdp = normalized_mutual_info_score(ground_truth_new, labels_sdp)
nmi_kmeans = normalized_mutual_info_score(ground_truth_new, kmeans_side_info)

print('kmeans only', nmi_kmeans)
print('sdp only on graph data', nmi_sdp)
print('spectral clustering (median) only', nmi_spectral)



sc_ = cluster.SpectralClustering(2, affinity='precomputed')
sc_.fit(adj_matrix_new)
sc_graph_score = normalized_mutual_info_score(ground_truth_new, sc_.labels_)
print("spectral clustering on graph data only", sc_graph_score)

h = spectral_embedding_side_info * 2 - 1
alpha_list_sp = [0.1, 1, 2, 5, 10, 30]
nmi_si_list_sp = []
for alpha in alpha_list_sp:
    labels_si = sdp2_si_B(G, alpha * h)
    nmi_si = normalized_mutual_info_score(ground_truth_new, labels_si)
    nmi_si_list_sp.append(nmi_si)



h = kmeans_side_info * 2 - 1
alpha_list = [1, 2, 5, 10, 30, 32]
nmi_si_list = []
for alpha in alpha_list:
    labels_si = sdp2_si_B(G, alpha * h)
    nmi_si = normalized_mutual_info_score(ground_truth_new, labels_si)
    nmi_si_list.append(nmi_si)



plt.plot(alpha_list, nmi_si_list, linewidth=1, marker='+', color='blue', label='KMeans')
plt.plot(alpha_list_sp, nmi_si_list_sp, linewidth=1, marker='x', color='red', label='SpectralEmbedding')
plt.xlabel('$\\xi$', size='large')
plt.ylabel("NMI", size='large')
plt.legend()
plt.savefig('sdp_si_friendship.pdf')
plt.show()


