import argparse
import networkx as nx
import numpy as np
from ising import SIBM_metropolis
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

def get_ground_truth(graph):
    g0 = set()
    g1 = set()
    for n in graph.nodes(data=True):
        if n[1]['block'] == 0:
            g0.add(n[0])
        else:
            g1.add(n[0])
    return (g0, g1)

def compare(r0, r1):
    '''
    get acc
    '''
    total_num = len(r0[0]) + len(r0[1])
    t0 = r0[0].intersection(r1[0])
    t1 = r0[1].intersection(r1[1])
    true_num = len(t0) + len(t1)
    t00 = r0[0].intersection(r1[1])
    t11 = r0[1].intersection(r1[0])
    true_num_2 = len(t00) + len(t11)
    if true_num > true_num_2:
        return true_num / total_num
    else:
        return true_num_2 / total_num

def get_acc(graph, alg, num_of_times=100):
    acc = 0
    for i in range(num_of_times):
        gt = get_ground_truth(graph)
        if alg == 'bisection':
            results = nx.algorithms.community.kernighan_lin.kernighan_lin_bisection(graph)
        elif alg == 'metropolis':
            results = SIBM_metropolis(graph)
        else:
            raise NotImplementedError('')
        acc += compare(gt, results)
    acc /= num_of_times
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', choices=['metropolis', 'bisection'], default='metropolis')
    parser.add_argument('--repeat', type=int, default=100, help='number of times to generate the SBM graph')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--a', type=float, default=16)
    parser.add_argument('--b', type=float, default=4)
    parser.add_argument('--draw', type=bool, const=True, nargs='?', default=False)
    args = parser.parse_args()
    graph = sbm_graph(args.n, args.a, args.b)
    if args.draw:
        draw(graph)
    print(get_acc(graph, args.alg, args.repeat))