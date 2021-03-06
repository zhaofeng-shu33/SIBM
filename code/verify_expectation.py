# verify \sum exp(beta * (B_i - A_i )) = n^{g(beta)}
import argparse
import logging
from multiprocessing import Queue
from multiprocessing import Process

import numpy as np
from matplotlib import pyplot as plt

from sbm import sbm_graph

def theoretical(n, a, b):
    beta_star = np.log((a + b -2 - np.sqrt((a + b - 2) ** 2 - 4 * a * b)) / (2 * b))
    beta = np.linspace(0, beta_star)
    coefficient = 1 - np.log(n) ** 2 / (4 * n) * (a ** 2 * (np.exp(-1 * beta) - 1) ** 2 + b ** 2 * (np.exp(beta) - 1) ** 2)
    plt.plot(beta, coefficient)
    plt.show()

def get_average(G, a, b, beta, filter_positive=True):
    n = len(G.nodes)
    L = np.zeros([n])
    for u, v in G.edges:
        if G.nodes[u]['block'] == G.nodes[v]['block']:
            L[u] -= 1
            L[v] -= 1
        else:
            L[u] += 1
            L[v] += 1
    if filter_positive and np.any(L >= 0):
        return 0
    val = np.sum(np.exp(beta * L))
    return val

def get_average_k(G, a, b, k, beta):
    # G should have multiple communities
    # \sum_{i=1}^n exp(beta * (A^{1}_i + A^{2}_i - 2 * A^{0}_i))
    n = len(G.nodes)
    L = np.zeros([n])
    kminus1 = k - 1
    for u, v in G.edges:
        if G.nodes[u]['block'] == G.nodes[v]['block']:
            L[u] -= kminus1
            L[v] -= kminus1
        else:
            L[u] += 1
            L[v] += 1
    val = np.sum(np.exp(beta * L))
    return val

def get_multiple_values(n, a, b, k, beta, repeat, action, qu):
    val_list = []
    for _ in range(repeat):
        if action == 'normal':
            G = sbm_graph(n, 2, a, b)
            val_list.append(get_average(G, a, b, beta))
        elif action == 'mixed':
            G = sbm_graph(n, k, a, b)
            val_list.append(get_average_k(G, a, b, k, beta))
    qu.put(val_list)

def g(x, a, b):
    tmp = (b * np.exp(x) + a * np.exp(-x)) / 2 - (a + b) / 2 + 1
    min_val = 1 - (np.sqrt(a) - np.sqrt(b)) ** 2 / 2
    return np.min([tmp, min_val])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['normal', 'mixed'], default='normal')
    parser.add_argument('--a', type=float, default=16.0)
    parser.add_argument('--b', type=float, default=4.0)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--repeat', type=int, default=1500, help='number of times to generate the SBM graph')
    parser.add_argument('--thread_num', type=int, default=1)
    args = parser.parse_args()
    val_list = []
    a = args.a
    b = args.b
    k = args.k
    beta = args.beta
    repeat = args.repeat
    thread_num = args.thread_num
    n = args.n    
    q = Queue()
    process_list = []
    assert(repeat % thread_num == 0)
    inner_repeat = repeat // thread_num
    for _ in range(thread_num):
        t = Process(target=get_multiple_values,
                    args=(n, a, b, k, beta, inner_repeat, args.action, q))
        process_list.append(t)
        t.start()
    total_val_list = []
    for i in range(thread_num):
        process_list[i].join()
        val_list = q.get()
        total_val_list.extend(val_list)
    if args.action == 'normal':
        mean_value = np.mean(total_val_list) / np.power(n, g(beta, a, b))
        var_value = np.var(total_val_list) / np.power(n, g(2 * beta, a, b))
        print(mean_value, var_value)
    elif args.action == 'mixed':
        print(np.mean(total_val_list))