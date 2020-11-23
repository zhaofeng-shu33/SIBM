'''
   parameter estimation validation for SSBM(n, k, p, q)
   with p = a log(n)/ n, q = b log(n) / n
   python3 estimator_experiment.py --action compute --repeat 1000 --thread_num 20
'''
import os
import pickle
import logging
import argparse
from datetime import datetime
from multiprocessing import Queue
from multiprocessing import Process

import matplotlib.pyplot as plt

from ising import estimate_a_b
from sbm import sbm_graph
from sbm import set_up_log
from sibm_experiment import save_data_to_pickle

def load_data_from_pickle(file_name_prefix, date):
    # save the data in pickle format
    file_name = file_name_prefix + '-' + date + '.pickle'
    with open(os.path.join('build', file_name), 'rb') as f:
        return pickle.load(f)

def plot_result(a_b_k_list, n_list, error_double_list):
    color_list = ['r', 'b']
    for i in range(len(a_b_k_list)):
        a, b, k = a_b_k_list[i]
        error_list = error_double_list[i]
        label_text = 'a={0},b={1},k={2}'.format(a, b, k)
        plt.plot(n_list, error_list, label=label_text, linewidth=4, color=color_list[i])
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
        G = sbm_graph(n, k, a, b)
        a_hat, b_hat = estimate_a_b(G, k)
        square_error += (a - a_hat) ** 2 + (b - b_hat) ** 2
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
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()
    n_list = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 10000]
    a_b_k_list = [(16, 4, 2), (16, 4, 4)]
    repeat = args.repeat
    thread_num = args.thread_num
    if args.action == 'compute':
        compute(n_list, a_b_k_list, repeat, thread_num)
    else:
        dic = load_data_from_pickle('estimator', args.date)
        error_double_list = dic['error_list']
        plot_result(a_b_k_list, n_list, error_double_list)
    
