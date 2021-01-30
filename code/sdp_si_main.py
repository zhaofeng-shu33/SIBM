import argparse
import logging
import os
import pickle
from multiprocessing import Process, Queue

try:
    import matplotlib.pyplot as plt
except:
    pass
import numpy as np

from sbm import sbm_graph, get_ground_truth, compare, set_up_log
from sbm import phase_transition_interval
from sdp import sdp2_si

def generate_data(ground_truth, n, m, p0, p1):
    # n, m = data.shape
    data = np.zeros([n, m], dtype=int)
    for i in range(n):
        if ground_truth[i] == 1:
            p = p0
        else:
            p = p1
        data[i, :] = np.random.choice([0, 1], size=m, p=p)
    return data

def get_acc(params, num_of_times=100, multi_thread=1):
    acc = 0
    q = Queue()
    if multi_thread == 1:
        acc_task(params, num_of_times, q)
        acc = q.get()
        acc /= num_of_times
    else:
        process_list = []        
        num_of_times_per_process = num_of_times // multi_thread
        for _ in range(multi_thread):
            t = Process(target=acc_task,
                        args=(params, num_of_times_per_process, q))
            process_list.append(t)
            t.start()
        for i in range(multi_thread):
            process_list[i].join()
            acc += q.get()
        acc /= (num_of_times_per_process * multi_thread)
    return acc

def acc_task(params, num_of_times, qu):
    acc = 0
    n, m, a, b, p0, p1 = params
    for _ in range(num_of_times):
        graph = sbm_graph(n, 2, a, b)
        gt = get_ground_truth(graph)
        a_b_ratio = a / b
        data = generate_data(gt, n, m, p0, p1)
        result_label = sdp2_si(graph, data, p0, p1, a_b_ratio, rho = 0.1, max_iter = 1000, tol=1e-4)
        current_acc = int(compare(gt, result_label))
        acc += current_acc
    qu.put(acc)

def recovery_matrix_simulation(n, m, p0, p1, a_range=[2, 25], b_range=[2, 10], step=0.5, repeat=20, multi_thread=20):
    b_start, b_end = b_range
    b_list = np.arange(b_start, b_end, step)
    b_num = len(b_list)
    a_start, a_end = a_range
    a_list = np.arange(a_start, a_end, step)
    a_num = len(a_list)
    Z = np.zeros([a_num, b_num])
    counter = 0
    total_points = a_num * b_num
    for i, b in enumerate(b_list):
        for j, a in enumerate(a_list):
            logging.info('finished %.2f' % (100 * counter / total_points) + '%')
            counter += 1
            if a <= b:
                continue
            acc = get_acc((n, m, a, b, p0, p1), repeat, multi_thread)
            Z[j, i] = acc
    prefix = 'p0p1-ab-{0}-{1}-{2}-{3}'.format(a_start, a_end, b_start, b_end)
    if os.environ.get('SLURM_ARRAY_TASK_ID'):
        prefix = 'slurm-{0}-'.format(os.environ['SLURM_ARRAY_TASK_ID']) + prefix
    phase_transition_interval(prefix, a_list, b_list, Z)

def simulation_plot(filename, n, m, p0, p1):
    with open(filename, 'rb') as f:
        dic = pickle.load(f)
    a_list = dic['a']
    b_list = dic['b']
    Z = dic['acc_list']
    b_num = len(b_list)
    b_min, b_max = b_list[0], b_list[b_num - 1]
    a_num = len(a_list)
    a_min, a_max = a_list[0], a_list[a_num - 1]

    plt.imshow(Z, cmap='Greys_r', origin='lower', extent=[b_min, b_max, a_min, a_max])
    x = np.linspace(b_min, b_max)
    y = (np.sqrt(2) + np.sqrt(x)) ** 2
    plt.xlabel('b')
    plt.ylabel('a')
    plt.plot(x, y, color='blue', label='sdp only')
    D12 = -2 * np.log(np.sqrt(p0 * p1) + np.sqrt((1 - p0) * (1 - p1)))
    gamma = m / np.log(n)
    y = (np.sqrt(2 - gamma * D12) + np.sqrt(x)) ** 2
    plt.plot(x, y, color='red', label='ours')
    plt.colorbar()
    _title = 'p0={0}, p1={1}, n={2}, m={3}'.format(p0, p1, n, m)
    plt.title(_title)
    plt.legend()
    plt.savefig('build/' + _title + '.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=1, help='number of times to generate the SBM graph')
    parser.add_argument('--multi_thread', type=int, default=1)
    parser.add_argument('--action', choices=['single', 'range', 'plot'], default='single')
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--a', type=float, default=[16.0], nargs='+')
    parser.add_argument('--b', type=float, default=[4.0], nargs='+')
    parser.add_argument('--p0', type=float, default=[0.8, 0.2], nargs='+')
    parser.add_argument('--p1', type=float, default=[0.2, 0.8], nargs='+')
    args = parser.parse_args()
    set_up_log()
    n = args.n
    m = args.m
    a = args.a
    b = args.b
    p0 = args.p0
    p1 = args.p1
    repeat = args.repeat
    multi_thread = args.multi_thread
    if args.action == 'single':
        a = a[0]
        b = b[0]
        acc = get_acc((n, m, a, b, p0, p1), repeat, multi_thread)
        print(acc)
    elif args.action == 'range':
        a_range = [a[0], a[1]]
        b_range = [b[0], b[1]]
        step = a[2]
        recovery_matrix_simulation(n, m, p0, p1, a_range, b_range, step, repeat=repeat, multi_thread=multi_thread)