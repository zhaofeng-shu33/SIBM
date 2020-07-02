import argparse
import networkx as nx
import numpy as np
def sbm_graph(n, a, b):
    if n % 2 != 0 or a <= b:
        raise ValueError('')
    sizes = [int(n/2), int(n/2)]
    _p = np.log(n) * a / n
    _q = np.log(n) * b / n
    if _p > 1 or _q > 1:
        raise ValueError('')
    p = [[_p, _q], [_q, _p]]
    return nx.generators.community.stochastic_block_model(sizes, p)

def draw(graph):
    ag = nx.nx_agraph.to_agraph(graph)
    ag.node_attr['shape'] = 'point'
    ag.edge_attr['style'] = 'dotted'
    ag.edge_attr['color'] = 'blue'
    ag.edge_attr['penwidth'] = 0.3
    # ag.write('build/a.dot')
    # weighting the edges according to block
    for e in ag.edges_iter():
        n1 = ag.get_node(e[0])
        n2 = ag.get_node(e[1])
        if n1.attr['block'] == n2.attr['block']:
            e.attr['weight'] = 10
            e.attr['len'] = 0.5
        else:
            e.attr['weight'] = 1
            e.attr['len'] = 1
    ag.write('build/a.dot')
    ag.layout()
    ag.draw('build/a.eps')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--a', type=float, default=10)
    parser.add_argument('--b', type=float, default=8)
    parser.add_argument('--draw', type=bool, const=True, nargs='?', default=False)
    args = parser.parse_args()
    graph = sbm_graph(args.n, args.a, args.b)
    if args.draw:
        draw(graph)
