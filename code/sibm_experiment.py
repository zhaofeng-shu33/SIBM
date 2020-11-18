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

has_cpp_wrapper = True
try:
    from sibm_c import task_cpp_wrapper
except ImportError:
    has_cpp_wrapper = False

def save_data_to_pickle(file_name_prefix, data):
    # save the data in pickle format
    file_name = file_name_prefix + '-' + datetime.now().strftime('%Y-%m-%d') + '.pickle'
    with open(os.path.join('build', file_name), 'wb') as f:
        pickle.dump(data, f)

def save_beta_transition_data(a, b, n, k, beta_list, acc_list):
    prefix = 'beta_trans_a_{0}_b_{1}_n_{2}_k_{3}'.format(a, b, n, k)
    data = {'a': a, 'b': b, 'n': n, 'k': k,
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

class SIBMk:
    '''SIBM with multiple community'''
    def __init__(self, graph, _alpha, _beta, k=2):
        self.G = graph
        self._beta = _beta
        self._alpha_divide_beta = _alpha / self._beta
        self.n = len(self.G.nodes)
        # randomly initiate a configuration
        self.sigma = [1 for i in range(self.n)]
        nodes = list(self.G)
        random.Random().shuffle(nodes)
        self.k = k
        for node_state in range(k):
            for i in range(self.n // k):
                self.sigma[nodes[i * k + node_state]] = node_state
        # node state is 0, 1, \dots, k-1
        self.m = [self.n / k for i in range(k)] # each number of w^s
        self.mixed_param = self._alpha_divide_beta * np.log(self.n)
        self.mixed_param /= self.n       
    def get_dH(self, trial_location, w_s):
        _sum = 0
        sigma_r = self.sigma[trial_location]
        w_s_sigma_r = (w_s + sigma_r) % self.k
        for i in self.G[trial_location]:
            if sigma_r == self.sigma[i]:
                _sum += 1
            elif w_s_sigma_r == self.sigma[i]:
                _sum -= 1
        _sum *= (1 + self.mixed_param)
        _sum += self.mixed_param * (self.m[w_s_sigma_r] - self.m[sigma_r] + 1)
        return _sum
    def _metropolis_single(self):
        # randomly select one position to inspect
        r = random.randint(0, self.n - 1)
         # randomly select one new flipping state to inspect
        w_s = random.randint(1, self.k - 1)
        delta_H = self.get_dH(r, w_s)
        if delta_H < 0:  # lower energy: flip for sure
            self.m[self.sigma[r]] -= 1
            self.sigma[r] = (w_s + self.sigma[r]) % self.k
            self.m[self.sigma[r]] += 1
        else:  # Higher energy: flip sometimes
            probability = np.exp(-1.0 * self._beta * delta_H)
            if np.random.rand() < probability:
                self.m[self.sigma[r]] -= 1
                self.sigma[r] = (w_s + self.sigma[r]) % self.k
                self.m[self.sigma[r]] += 1
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

def exact_compare_k(labels, k):
    n = len(labels)
    nk = n // k
    labels_inner = np.array(labels)
    for i in range(k):
        candidate = labels[nk * i]
        if np.any(labels_inner[nk * i : nk * (i + 1)] != candidate):
            return False
    return True
def majority_voting(labels_list):
    # all samples align with the first
    labels_np = np.array(labels_list, dtype=int)
    m = len(labels_list)
    for i in range(1, m):
        if np.dot(labels_np[0, :], labels_np[i, :]) < 0:
            labels_np[i, :] *= -1
    voting_result = np.sum(labels_np, axis=0) > 0
    return 2 * np.asarray(voting_result, dtype=int) - 1

def task(repeat, n, k, a, b, alpha, beta, num_of_sibm_samples, m, _N, qu=None):
    if k == 2 and m == 1 and has_cpp_wrapper:
        total_acc = task_cpp_wrapper(repeat, n, a, b, alpha, beta, num_of_sibm_samples, _N)
        if qu is not None:
            qu.put(total_acc)
            return
        else:
            return total_acc
    total_acc = 0
    for _ in range(repeat):
        G = sbm_graph(n, k, a, b)
        if k == 2:
            sibm = SIBM2(G, alpha, beta)
        else:
            sibm = SIBMk(G, alpha, beta, k)
        sibm.metropolis(N=_N)
        acc = 0
        if m > 1:
            sample_list = []
            for _ in range(num_of_sibm_samples * m):
                sibm._metropolis_single()
                sample_list.append(sibm.sigma.copy())
            for i in range(num_of_sibm_samples):
                # collect nearly independent samples
                candidates_samples = [sample_list[i + j * num_of_sibm_samples] for j in range(m)]
                inner_acc = int(exact_compare(majority_voting(candidates_samples))) # for exact recovery
        else:
            for _ in range(num_of_sibm_samples):
                sibm._metropolis_single()
                if k == 2:
                    inner_acc = int(exact_compare(sibm.sigma))
                else:
                    inner_acc = int(exact_compare_k(sibm.sigma, k))
        acc += inner_acc
        acc /= num_of_sibm_samples
        total_acc += acc
    total_acc /= repeat
    if qu is not None:
        qu.put(total_acc)
        return
    else:
        return total_acc

def get_acc(repeat, n, k, a, b, alpha, beta, num_of_sibm_samples, m, _N, thread_num):
    if thread_num == 1:
        return task(repeat, n, k, a, b, alpha, beta, num_of_sibm_samples, m, _N)
    q = Queue()
    process_list = []
    assert(repeat % thread_num == 0)
    inner_repeat = repeat // thread_num
    for _ in range(thread_num):
        t = Process(target=task,
                    args=(inner_repeat, n, k, a, b, alpha, beta, num_of_sibm_samples, m, _N, q))
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
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--beta', type=float, default=1.0, nargs='+')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to generate the SBM graph')
    parser.add_argument('--inner_repeat', type=int, default=1, help='number of sigma generated for a given graph')
    parser.add_argument('--m', type=int, default=1, help='samples used for majority voting, odd number')
    parser.add_argument('--max_iter', type=int, default=100, help='burn-in period')
    parser.add_argument('--disable_c_binding', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--thread_num', type=int, default=1)
    args = parser.parse_args()
    set_up_log()
    if args.disable_c_binding:
        has_cpp_wrapper = False
    if type(args.beta) is float:
        beta_list = [args.beta]
    elif len(args.beta) == 3 and args.beta[-1] < args.beta[-2]:
        # start, end, step
        beta_list = np.arange(args.beta[0], args.beta[1], args.beta[2])
    else:
        beta_list = args.beta
    acc_list = []
    for beta in beta_list:
        averaged_acc = get_acc(args.repeat, args.n, args.k, args.a, args.b,
                            args.alpha, beta, args.inner_repeat, args.m,
                            args.max_iter, args.thread_num)
        acc_list.append(averaged_acc)
        logging.info('a: {0}, b: {1}, n: {2}, beta: {3}, acc: {4} '.format(args.a, args.b, args.n, beta, averaged_acc))
    save_beta_transition_data(args.a, args.b, args.n, args.k, beta_list, acc_list)
