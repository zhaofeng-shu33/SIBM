import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

def plot_alg_fix_b(alg_list, date):
    for alg in alg_list:
        file_name = alg + '-transition-' + date + '.pickle'
        with open(os.path.join('build', file_name), 'rb') as f:
            data = pickle.load(f)
            plt.plot(data['a'], data['acc_list'], label=alg, linewidth=4)
    plt.legend()
    plt.xlabel('a', size='large')
    plt.ylabel('score', size='large')
    fig_name = 'transition-b3-' + date + '.svg'
    plt.savefig(os.path.join('build', fig_name), transparent=True)
    plt.show()

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
    plt.plot(x, y, color='red')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['phase_transition',
        'compare'], default='phase_transition')
    args = parser.parse_args()
    if args.action == 'phase_transition':
        file_name = 'transition-2020-09-14.pickle'
        draw_phase_transation(file_name)
    else:
        plot_alg_fix_b(['sdp', 'metropolis', 'asyn_fluid'],
            '2020-09-16')