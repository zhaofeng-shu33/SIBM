import argparse
from multiprocessing import Queue
from datetime import datetime
import logging
import os
import pickle

import networkx as nx
import numpy as np
from sklearn import metrics
from ising import SIBM_metropolis, SIBM
from sdp import sdp, sdp2, solve_sdp_cvx

def set_up_log():
    LOGGING_FILE = 'simulation.log'
    logFormatter = logging.Formatter('%(asctime)s %(message)s')
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join('build', LOGGING_FILE))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    if os.environ.get("LOGLEVEL"):
        rootLogger.setLevel(os.environ.get("LOGLEVEL"))
    else:
        rootLogger.setLevel(logging.INFO)

def sbm_graph(n, k, a, b):
    if n % k != 0:
        raise ValueError('n %k != 0')
    elif a <= b:
        raise ValueError('a <= b')
    sizes = [int(n/k) for _ in range(k)]
    _p = np.log(n) * a / n
    _q = np.log(n) * b / n
    if _p > 1 or _q > 1:
        raise ValueError('%f (probability) larger than 1' % _p)
    p = np.diag(np.ones(k) * (_p - _q)) + _q * np.ones([k, k])
    return nx.generators.community.stochastic_block_model(sizes, p)

def draw_metropolis():
    ''' demo for Metropolis sampling, run this function with
        python3 -c "from sbm import draw_metropolis; draw_metropolis()"
    '''
    n = 120
    K = 3
    a = 9
    b = 1
    graph = sbm_graph(n, K, a, b)
    results = SIBM_metropolis(graph, k=K)
    for i in range(n):
        graph.nodes[i]['estimated_block'] = results[i]
    draw(graph)

def get_agraph(graph, node_color_index=None):
    '''
    Parameters
    ----------
    graph: networkx graph object
    '''
    has_estimated_block = graph.nodes[0].get('estimated_block') is not None
    # Graphviz AGraph
    ag = nx.nx_agraph.to_agraph(graph)
    ag.graph_attr['outputorder'] = 'edgesfirst'
    ag.graph_attr['bgcolor'] = 'transparent'
    ag.node_attr['shape'] = 'point'
    # ag.edge_attr['style'] = 'dotted'
    ag.edge_attr['color'] = 'gray'
    ag.edge_attr['penwidth'] = 0.3
    # coloring the node according to estimator block
    color_list = ['red', 'blue', 'green']
    if has_estimated_block:
        for n in ag.nodes_iter():
            n.attr['color'] = color_list[int(n.attr['estimated_block'])]
    elif node_color_index is not None:
        for n in ag.nodes_iter():
            n.attr['color'] = color_list[node_color_index[int(str(n))]]
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
    return ag

def draw(graph):
    ag = get_agraph(graph)
    ag.write('build/a.dot')
    ag.layout()
    ag.draw('build/a.eps')

def get_ground_truth(graph):
    label_list = []
    for n in graph.nodes(data=True):
        label_list.append(n[1]['block'])
    return label_list

def compare(label_0, label_1):
    '''
    get acc using adjusted rand index
    '''
    return metrics.adjusted_rand_score(label_0, label_1)

def convert_to_label_list(n, partition):
    cat = [0 for i in range(n)]
    label_index = 0
    for i in partition:
        for j in i:
            cat[j] = label_index
        label_index += 1
    return cat

def acc_task(alg, params, num_of_times, qu):
    acc = 0
    n, k, a, b, binary = params
    for _ in range(num_of_times):
        graph = sbm_graph(n, k, a, b)
        gt = get_ground_truth(graph)
        if alg == 'bisection':
            results_partition = nx.algorithms.community.kernighan_lin.kernighan_lin_bisection(graph)
            results = convert_to_label_list(n, results_partition)
        elif alg == 'modularity':
            results_partition = nx.algorithms.community.modularity_max.greedy_modularity_communities(graph)
            results = convert_to_label_list(n, results_partition)
        elif alg == 'asyn_fluid':
            results_partition = nx.algorithms.community.asyn_fluid.asyn_fluidc(graph, k)
            results = convert_to_label_list(n, results_partition)
        elif alg == 'sdpk':
            results = sdp(graph, k)
        elif alg == 'sdp2':
            results = sdp2(graph)
        elif alg == 'sdp_ip':
            results = solve_sdp_cvx(graph)
        elif alg == 'metropolis':
            if logging.getLogger().level == logging.DEBUG:
                sibm = SIBM(graph, k)
                results = sibm.metropolis(N=40)
                h_value = sibm._get_Hamiltonian()
                logging.debug(results)
                logging.debug(h_value)
            else:
                results = SIBM_metropolis(graph, k)
        else:
            raise NotImplementedError('')
        current_acc = compare(gt, results)
        if binary:
            current_acc = int(current_acc)
        acc += current_acc
    qu.put(acc)

def get_acc(alg, params, num_of_times=100, multi_thread=1):
    acc = 0
    q = Queue()
    if multi_thread == 1:
        acc_task(alg, params, num_of_times, q)
        acc = q.get()
        acc /= num_of_times
    else:
        from multiprocessing import Process
        process_list = []        
        num_of_times_per_process = num_of_times // multi_thread
        for _ in range(multi_thread):
            t = Process(target=acc_task,
                        args=(alg, params, num_of_times_per_process, q))
            process_list.append(t)
            t.start()
        for i in range(multi_thread):
            process_list[i].join()
            acc += q.get()
        acc /= (num_of_times_per_process * multi_thread)
    return acc

def phase_transition_interval(file_name_prefix, a_list, b_list, acc_list):
    # save the data in pickle format
    data = {'a': a_list, 'b': b_list, 'acc_list': acc_list}
    file_name = file_name_prefix + '-' + datetime.now().strftime('transition-%Y-%m-%d') + '.pickle'
    with open(os.path.join('build', file_name), 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # sdpk: sdp for k community, wrapper for corresponding R package
    # sdp2: sdp for 2 community, ADMM implementation
    # sdp_ip, sdp using cvx toolbox, interior point implementation
    parser.add_argument('--alg', choices=['metropolis', 'bisection', 'asyn_fluid', 'modularity', 'sdpk', 'sdp2', 'sdp_ip'], default='metropolis')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to generate the SBM graph')
    parser.add_argument('--multi_thread', type=int, default=1)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--binary', type=bool, const=True, nargs='?', default=False)
    parser.add_argument('--a', type=float, default=16.0, nargs='+')
    parser.add_argument('--b', type=float, default=4.0, nargs='+')
    parser.add_argument('--draw', type=bool, const=True, nargs='?', default=False)
    args = parser.parse_args()
    set_up_log()
    if type(args.a) is float:
        args.a = [args.a]
    elif len(args.a) == 3 and args.a[-1] < args.a[-2]:
        # start, end, step
        args.a = np.arange(args.a[0], args.a[1], args.a[2])

    if type(args.b) is float:
        args.b = [args.b]
    elif len(args.b) == 3 and args.b[-1] < args.b[-2]:
        args.b = np.arange(args.b[0], args.b[1], args.b[2])

    acc_list = []
    total_points = len(args.a) * len(args.b)
    counter = 0
    for a in args.a:
        for b in args.b:
            counter += 1
            if a <= b:
                continue
            params = (args.n, args.k, a, b, args.binary)
            acc = get_acc(args.alg, params, args.repeat,
                        multi_thread=args.multi_thread)
            acc_list.append(acc)
            if counter % 10 == 0:
                logging.info('finished %.2f' % (100 * counter / total_points) + '%')
            else:
                logging.debug('finished %.2f' % (100 * counter / total_points) + '%')
    logging.info('n: {0}, repeat: {1}, alg: {2}'.format(args.n, args.repeat, args.alg))
    if len(acc_list) > 1:
        file_name_prefix = args.alg
        phase_transition_interval(file_name_prefix, args.a, args.b, acc_list)
    if args.draw and len(args.a) > 1:
        from matplotlib import pyplot as plt
        plt.plot(args.a, acc_list)
        plt.savefig('build/%s.png' % datetime.now().strftime('acc-a-%H-%M-%S'))
    if len(acc_list) == 1:
        logging.info('a: {0}, b: {1}, acc: {2}'.format(args.a[0], args.b[0], acc_list[0]))