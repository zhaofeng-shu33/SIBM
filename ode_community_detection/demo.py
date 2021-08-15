import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from kuramoto import Kuramoto
from correlation import correlation_from_adj
from sbm import sbm_graph
# Instantiate a random graph and transform into an adjacency matrix
graph_nx = sbm_graph(100, 2, 16, 2)
graph = nx.to_numpy_array(graph_nx)
n = len(graph)
dt = 0.05
# Instantiate model with parameters
natfreqs = np.zeros([n])
model = Kuramoto(coupling=0.3, dt=dt, T=5.0, n_nodes=n, natfreqs=natfreqs)

# Run simulation - output is time series for all nodes (node vs time)
act_mat = model.run(adj_mat=graph)
rho = correlation_from_adj(act_mat, dt)
# Plot all the time series
#plt.plot(rho[1, 71,:])
label_list = [int(i) for i in range(10)]
plt.figure(0)
plt.plot(rho[0,1:100:10,:].T, label=label_list)
plt.legend()
plt.matshow(rho[:, :, -1])
plt.show()