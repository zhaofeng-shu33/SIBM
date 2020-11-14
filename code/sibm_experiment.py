import argparse
import logging
import pickle
import random
import os
from datetime import datetime
from multiprocessing import Queue
from multiprocessing import Process

import numpy as np

from sbm import sbm_graph
from sbm import set_up_log
from sbm import compare, get_ground_truth

def save_data_to_pickle(file_name_prefix, data):
    # save the data in pickle format
    file_name = file_name_prefix + '-' + datetime.now().strftime('%Y-%m-%d') + '.pickle'
    with open(os.path.join('build', file_name), 'wb') as f:
        pickle.dump(data, f)

def save_beta_transition_data(a, b, n, beta_list, acc_list):
    prefix = 'beta_trans'
    data = {'a': a, 'b': b, 'n': n,
            'beta_list': beta_list, 'acc_list': acc_list
           }
    save_data_to_pickle(prefix, data)

class SIBM2:
    '''SIBM with two community'''
    def __init__(self, graph, _alpha, _beta):
        self.G = graph
        self._beta = _beta
        self._alpha_divide_beta = _alpha / self._beta
        self.n = len(self.G.nodes)
        # randomly initiate a configuration
        self.sigma = [1 for i in range(self.n)]
        nodes = list(self.G)
        random.Random().shuffle(nodes)
        k = 2
        self.k = k
        for node_state in range(k):
            for i in range(self.n // k):
                self.sigma[nodes[i * k + node_state]] = 2 * node_state - 1
        # node state is +1 or -1
        self.m = self.n / 2 # number of +1
        self.mixed_param = self._alpha_divide_beta * np.log(self.n)
        self.mixed_param /= self.n
    def get_dH(self, trial_location):
        _sum = 0
        w_s = self.sigma[trial_location]
        for i in self.G[trial_location]:
                _sum += self.sigma[i]
        _sum *= w_s * (1 + self.mixed_param)
        _sum -= self.mixed_param * (w_s * (2 * self.m - self.n) - 1)
        return _sum
    def _metropolis_single(self):
        # randomly select one position to inspect
        r = random.randint(0, self.n - 1)
        delta_H = self.get_dH(r)
        if delta_H < 0:  # lower energy: flip for sure
            self.sigma[r] *= -1
            self.m += self.sigma[r]
        else:  # Higher energy: flip sometimes
            probability = np.exp(-1.0 * self._beta * delta_H)
            if np.random.rand() < probability:
                self.sigma[r] *= -1
                self.m += self.sigma[r]
    def metropolis(self, N=40):
        # iterate given rounds
        for _ in range(N):
            for _ in range(self.n):
                self._metropolis_single()
        return self.sigma

def exact_compare(labels):
    # return 1 if labels = X or -X
    n2 = int(len(labels) / 2)
    labels_inner = np.array(labels)
    return np.sum(labels_inner) == 0 and np.abs(np.sum(labels_inner[:n2])) == n2

def majority_voting(labels_list):
    # all samples align with the first
    labels_np = np.array(labels_list, dtype=int)
    m = len(labels_list)
    for i in range(1, m):
        if np.dot(labels_np[0, :], labels_np[i, :]) < 0:
            labels_np[i, :] *= -1
    voting_result = np.sum(labels_np, axis=0) > 0
    return 2 * np.asarray(voting_result, dtype=int) - 1

def task(repeat, n, a, b, alpha, beta, num_of_sibm_samples, m, _N, qu):
    total_acc = 0
    for _ in range(repeat):
        G = sbm_graph(n, 2, a, b)
        sibm = SIBM2(G, alpha, beta)
        sibm.metropolis(N=_N)
        acc = 0
        sample_list = []
        for _ in range(num_of_sibm_samples * m):
            sibm._metropolis_single()
            sample_list.append(sibm.sigma.copy())
        for i in range(num_of_sibm_samples):
            # collect nearly independent samples
            candidates_samples = [sample_list[i + j * num_of_sibm_samples] for j in range(m)]
            inner_acc = int(exact_compare(majority_voting(candidates_samples))) # for exact recovery
            acc += inner_acc
        acc /= num_of_sibm_samples
        total_acc += acc
    total_acc /= repeat
    qu.put(total_acc)

def get_acc(repeat, n, a, b, alpha, beta, num_of_sibm_samples, m, _N, thread_num):
    q = Queue()
    process_list = []
    assert(repeat % thread_num == 0)
    inner_repeat = repeat // thread_num
    for _ in range(thread_num):
        t = Process(target=task,
                    args=(inner_repeat, n, a, b, alpha, beta, num_of_sibm_samples, m, _N, q))
        process_list.append(t)
        t.start()
    acc = 0
    for i in range(thread_num):
        process_list[i].join()
        acc += q.get()
    acc /= thread_num
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=16.0)
    parser.add_argument('--b', type=float, default=4.0)
    parser.add_argument('--n', type=int, default=300)
    # parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--beta', type=float, default=1.0, nargs='+')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to generate the SBM graph')
    parser.add_argument('--inner_repeat', type=int, default=1, help='number of sigma generated for a given graph')
    parser.add_argument('--m', type=int, default=1, help='samples used for majority voting, odd number')
    parser.add_argument('--max_iter', type=int, default=100, help='burn-in period')
    parser.add_argument('--thread_num', type=int, default=1)
    args = parser.parse_args()
    set_up_log()
    if type(args.beta) is float:
        beta_list = [args.beta]
    elif len(args.beta) == 3 and args.beta[-1] < args.beta[-2]:
        # start, end, step
        beta_list = np.arange(args.beta[0], args.beta[1], args.beta[2])
    else:
        beta_list = args.beta
    acc_list = []
    for beta in beta_list:
        averaged_acc = get_acc(args.repeat, args.n, args.a, args.b,
                            args.alpha, beta, args.inner_repeat, args.m,
                            args.max_iter, args.thread_num)
        acc_list.append(averaged_acc)
        logging.info('a: {0}, b: {1}, n: {2}, beta: {3}, acc: {4} '.format(args.a, args.b, args.n, beta, averaged_acc))
    save_beta_transition_data(args.a, args.b, args.n, beta_list, acc_list)
