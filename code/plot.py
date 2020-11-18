import os
import pickle
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

def plot_alg_fix_b(alg_list, date):
    '''compare different algorithms
    '''
    for alg in alg_list:
        file_name = alg + '-transition-' + date + '.pickle'
        f = open(os.path.join('build', file_name), 'rb')
        data = pickle.load(f)
        plt.plot(data['a'], data['acc_list'], label=alg, linewidth=4)
    plt.legend()
    plt.xlabel('a', size='large')
    plt.ylabel('score', size='large')
    fig_name = 'transition-b3-' + date + '.svg'
    plt.savefig(os.path.join('build', fig_name), transparent=True)
    plt.show()

def compute_empirical_beta(acc_list, beta_list, k=2):
    start_index = 0
    y_value = 1.0 / k
    for i in range(len(acc_list)):
        if acc_list[i] < y_value and acc_list[i + 1] > y_value:
            start_index = i
            break
    x_1 = beta_list[start_index]
    x_2 = beta_list[start_index + 1]
    y_1 = acc_list[start_index]
    y_2 = acc_list[start_index + 1]
    slope = (y_2 - y_1) / (x_2 - x_1)
    beta = x_2 - (y_2 - y_value) / slope
    return beta

def get_file_name_list(keyword_1, keyword_2):
    Ls = []
    for i in os.listdir('build'):
        if i.find(keyword_1) >= 0 and i.find(keyword_2) >= 0:
            Ls.append(i)
    return Ls

def draw_beta_phase_trans(date, pic_format='eps', theoretical=False):
    file_name_list = get_file_name_list('beta_trans', date + '.pickle')
    largest_n = 0
    for file_name in file_name_list:
        f = open(os.path.join('build', file_name), 'rb')
        data = pickle.load(f)
        if data['n'] > largest_n:
            largest_n = data['n']
        label_str = 'a = %.0f, b=%.0f, n=%d, k=%d' % (data['a'], data['b'], data['n'], data['k'])
        beta_list = data['beta_list']
        acc_list = data['acc_list']
        beta_star_empirical = compute_empirical_beta(acc_list, beta_list, data['k'])
        plt.plot(beta_list, acc_list, label=label_str, linewidth=1)
        plt.scatter([beta_star_empirical], [1.0 / data['k']], c='red')
    if theoretical:
        draw_theoretical_beta_phase_trans(largest_n, data['k'], data['a'], data['b'], beta_list[0], beta_list[-1])
    plt.legend()
    plt.xlabel('$beta$', size='large')
    plt.ylabel('acc', size='large')
    fig_name = 'beta_trans-' + date + '.' + pic_format
    plt.savefig(os.path.join('build', fig_name), transparent=True)
    plt.show()

    
def draw_theoretical_beta_phase_trans(n, k, a, b, beta_s, beta_e):
    # turning point
    beta_bar = 0.5 * np.log(a / b)
    # zero point
    beta_star = np.log((a + b - k - np.sqrt((a + b - k) ** 2 - 4 * a * b)) / (2 * b))
    g = lambda x: (b * np.exp(x) + a * np.exp(-x)) / k - (a + b) / k + 1
    g_beta_bar = g(beta_bar)
    beta_list_1 = np.linspace(beta_s, beta_star, num=50)
    beta_list_2 = np.linspace(beta_star, beta_e, num=50)
    acc_list_1 = np.zeros([50])
    acc_list_2 = np.zeros([50])
    for i, beta in enumerate(beta_list_1):
        cabdidate_1 = np.power(n, -1 * g(beta))
        acc_list_1[i] = np.max([cabdidate_1, np.power(n, g_beta_bar)])
    for i, beta in enumerate(beta_list_2):
        if beta < beta_bar:
            acc_list_2[i] = 1 - np.power(n, g(beta) / 2)
        else:
            acc_list_2[i] = 1 - np.power(n, g_beta_bar / 2)
    plt.plot(beta_list_1, acc_list_1)
    plt.plot(beta_list_2, acc_list_2)
    plt.plot([beta_star, beta_star], [0, 1])

def draw_phase_transation(file_name):
    with open(os.path.join('build', file_name), 'rb') as f:
        data = pickle.load(f)
    a_list = data['a']
    a_min = a_list[0]
    a_max =a_list[-1]
    b_list = data['b']
    b_min = b_list[0]
    b_max = b_list[-1]
    acc_list = data['acc_list']
    matrix = np.zeros([len(a_list), len(b_list)])
    counter = 0
    for i in range(len(a_list)):
        for j in range(len(b_list)):
            if a_list[i] <= b_list[j]:
                continue
            matrix[i, j] = acc_list[counter]
            counter += 1
    plt.imshow(matrix, cmap='Greys_r', origin='lower', extent=[b_min, b_max, a_min, a_max])
    x = np.linspace(b_min, b_max)
    y = (np.sqrt(2) + np.sqrt(x)) ** 2
    plt.xlabel('b')
    plt.ylabel('a')
    plt.plot(x, y, color='red')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    method_list = ['sdp', 'metropolis', 'asyn_fluid', 'bi', 'sdp2']
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['phase_transition',
        'compare', 'beta_transition'], default='phase_transition')
    parser.add_argument('--method', choices=method_list, default='metropolis')
    parser.add_argument('--format', choices=['eps', 'svg'], default='eps')
    parser.add_argument('--theoretical', const=True, default=False, nargs='?')
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()
    if args.action == 'phase_transition':
        file_name = args.method + '-transition-%s.pickle' % args.date
        draw_phase_transation(file_name)
    elif args.action == 'beta_transition':
        draw_beta_phase_trans(args.date, args.format, args.theoretical)
    else:
        plot_alg_fix_b(method_list, args.date)
