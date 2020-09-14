import networkx as nx

from ising import SIBM

def sample_graph():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(4, 5)
    G.add_edge(4, 6)
    G.add_edge(4, 7)
    G.add_edge(4, 8)
    G.add_edge(5, 7)
    G.add_edge(6, 7)
    G.add_edge(6, 8)
    G.add_edge(9, 10)
    G.add_edge(9, 11)
    G.add_edge(9, 12)
    G.add_edge(9, 13)
    G.add_edge(10, 11)
    G.add_edge(10, 13)
    G.add_edge(11, 12)
    G.add_edge(12, 13)
    G.add_edge(1, 5)
    G.add_edge(1, 9)
    G.add_edge(6, 10)
    G.add_edge(7, 11)
    return G

if __name__ == '__main__':
    # demo illustration to use sibm method for general graph
    G = sample_graph()
    sibm = SIBM(G, k=3, estimate_a_b=False, epsilon=0.1)
    print(sibm.metropolis())