"""
estimation error of maximum likelihood estimator for SSBM(n, 2, p, q)
with p = a log(n)/ n, q = b log(n) / n
"""

import os
import pickle
import logging
import argparse
from datetime import datetime
from multiprocessing import Queue
from multiprocessing import Process

import numpy as np
import matplotlib.pyplot as plt

from sbmising import SIBM, sbm_graph
from utility import set_up_log, save_data_to_pickle, get_ground_truth, compare

def load_data_from_pickle(file_name_prefix, date):
    # save the data in pickle format
    file_name = file_name_prefix + '-' + date + '.pickle'
    with open(os.path.join('build', file_name), 'rb') as f:
        return pickle.load(f)

def plot_result(a_b_k_list, n_list, error_double_list, theoretical=False):
    color_list = ['r', 'b', 'g']
    for i in range(len(a_b_k_list)):
        a, b, k = a_b_k_list[i]
        if theoretical and k == 2:
            linewidth = 1
            error_list_t = []
            for n in n_list:
                error_list_t.append(np.power(n, 1 - (np.sqrt(a) - np.sqrt(b)) ** 2 / 2))
            plt.plot(n_list, error_list_t, label='ML error', linewidth=linewidth, color=color_list[i], linestyle='dashed')
        else:
            linewidth = 4
        error_list = error_double_list[i]
        label_text = 'a={0},b={1},k={2}'.format(a, b, k)
        plt.plot(n_list, error_list, label=label_text, linewidth=linewidth, color=color_list[i])
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('n', size='large')
    plt.ylabel('square error', size='large')
    date = datetime.now().strftime('%Y-%m-%d')
    fig_name = 'estimator-error-' + date + '.eps'
    plt.savefig(os.path.join('build', fig_name), transparent=True)
    plt.show()

def estimator_once(n, k, a, b, repeat):
    square_error = 0
    for _ in range(repeat):
        G = sbm_graph(n, 2, a, b)
        labels = SIBM(G, k=2, max_iter=n, beta=np.log(a / b) / 2, gamma = 2 * b)
        labels_true = get_ground_truth(G)
        square_error += 1 - int(compare(labels, labels_true))
    square_error /= repeat
    logging.info('n: {0}, k: {1}, a: {2}, b: {3}, error: {4}'.format(n, k, a, b, square_error))
    return square_error

def estimator_multiple(n_list, k, a, b, repeat, thread_num, qu):
    square_error_list = []
    for n in n_list:
        square_error = estimator_once(n, k, a, b, repeat)
        square_error_list.append(square_error)
    qu.put(square_error_list)

def compute(n_list, a_b_k_list, repeat, thread_num):
    error_double_list = []
    index = 0
    for a, b, k in a_b_k_list:
        q = Queue()
        process_list = []
        assert(repeat % thread_num == 0)
        inner_repeat = repeat // thread_num
        for _ in range(thread_num):
            t = Process(target=estimator_multiple,
                        args=(n_list, k, a, b, inner_repeat, thread_num, q))
            process_list.append(t)
            t.start()
        for i in range(thread_num):
            process_list[i].join()
            error_list = q.get()
            if i == 0:
                error_double_list.append(error_list)
            else:
                for j in range(len(n_list)):
                    error_double_list[index][j] = (i * error_double_list[index][j] + error_list[j]) / (i + 1)
        index += 1

    dic = {'a_b_k_list': a_b_k_list, 'error_list': error_double_list, 'n_list': n_list}
    save_data_to_pickle('estimator', dic)
    
if __name__ == '__main__':
    set_up_log()
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['plot',
        'compute'], default='compute')
    parser.add_argument('--repeat', type=int, default=1000, help='number of times to generate the SBM graph')
    parser.add_argument('--n_list', type=int, default=[100, 200, 300], nargs='+')
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--theoretical', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()
    n_list = args.n_list
    a_b_k_list = [(16, 4, 2)]
    repeat = args.repeat
    thread_num = args.thread_num
    if args.action == 'compute':
        compute(n_list, a_b_k_list, repeat, thread_num)
    else:
        dic = load_data_from_pickle('estimator', args.date)
        error_double_list = dic['error_list']
        n_list = dic['n_list']
        plot_result(a_b_k_list, n_list, error_double_list, args.theoretical)
    
