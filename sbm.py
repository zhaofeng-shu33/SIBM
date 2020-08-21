import argparse
from multiprocessing import Queue
from datetime import datetime
import logging
import os

import networkx as nx
import numpy as np
from sklearn import metrics
from ising import SIBM_metropolis

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
    if n % k != 0 or a <= b:
        raise ValueError('')
    sizes = [int(n/k) for _ in range(k)]
    _p = np.log(n) * a / n
    _q = np.log(n) * b / n
    if _p > 1 or _q > 1:
        raise ValueError('')
    p = np.diag(np.ones(k) * (_p - _q)) + _q * np.ones([k, k])
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
    n, k, a, b = params
    for _ in range(num_of_times):
        graph = sbm_graph(n, k, a, b)
        gt = get_ground_truth(graph)
        if alg == 'bisection':
            results_partition = nx.algorithms.community.kernighan_lin.kernighan_lin_bisection(graph)
            results = convert_to_label_list(n, results_partition)
        elif alg == 'modularity':
            results_partition = nx.algorithms.community.modularity_max.greedy_modularity_communities(graph)
            results = convert_to_label_list(n, results_partition)
        elif alg == 'metropolis':
            results = SIBM_metropolis(graph, k)
            logging.debug(results)
        else:
            raise NotImplementedError('')
        acc += compare(gt, results)
    qu.put(acc)

def get_acc(alg, params, num_of_times=100, multi_thread=1, binary=False):
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
    if binary:
        acc = int(acc)
    return acc
def get_phase_transition_interval(a_list, acc_list):
    for i in range(len(acc_list)):
        if acc_list[i] < 0.5 and acc_list[i + 1] > 0.5:
            return (a_list[i], a_list[i + 1])
    raise ValueError('no interval found')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', choices=['metropolis', 'bisection', 'modularity'], default='metropolis')
    parser.add_argument('--repeat', type=int, default=100, help='number of times to generate the SBM graph')
    parser.add_argument('--multi_thread', type=int, default=1)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--binary', type=bool, const=True, nargs='?', default=False)
    parser.add_argument('--a', type=float, default=16.0, nargs='+')
    parser.add_argument('--b', type=float, default=4.0)
    parser.add_argument('--draw', type=bool, const=True, nargs='?', default=False)
    args = parser.parse_args()
    set_up_log()
    if type(args.a) is float:
        args.a = [args.a]
    acc_list = []
    for a in args.a:
        params = (args.n, args.k, a, args.b)
        acc = get_acc(args.alg, params, args.repeat,
                    multi_thread=args.multi_thread, binary=args.binary)
        acc_list.append(acc)

    logging.info('n: {0}, repeat: {1}, alg: {2}'.format(args.n, args.repeat, args.alg))
    if len(acc_list) > 1:
        a_left, a_right = get_phase_transition_interval(args.a, acc_list)
        logging.info('a_L: {0} for acc=0, a_R: {1} for acc=1, b: {2}'.format(a_left, a_right, args.b))
    if args.draw and len(args.a) > 1:
        from matplotlib import pyplot as plt
        plt.plot(args.a, acc_list)
        plt.savefig('build/%s.png' % datetime.now().strftime('acc-a-%H-%M-%S'))
    if len(acc_list) == 1:
        logging.info('a: {0}, b: {1}, acc: {2}'.format(args.a[0], args.b, acc_list[0]))