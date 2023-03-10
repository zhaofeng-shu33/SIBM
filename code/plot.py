import os
import pickle
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage

from sbm import sbm_graph, get_agraph
from ising import SIBM

def plot_g_function(a, b, k, end_point=None, fig_format='svg'):
    g = lambda x: (b * np.exp(x) + a * np.exp(-x)) / k - (a + b) / k + 1
    beta_bar = 0.5 * np.log(a / b)
    # zero point
    beta_star = np.log((a + b - k - np.sqrt((a + b - k) ** 2 - 4 * a * b)) / (2 * b))
    beta_2 = np.log((a + b - k + np.sqrt((a + b - k) ** 2 - 4 * a * b)) / (2 * b))
    if end_point is None:
        end_point = beta_2 * 1.2
    x = np.linspace(0, end_point)
    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.plot(x, g(x), label='$g(\\beta)$', color='red', linewidth=2)
    x2 = np.zeros([51])
    x2[:50] = np.linspace(0, beta_bar, num=50)
    y2 = g(x2) + 0.02
    x2[50] = end_point
    y2[50] = g(beta_bar)
    ax.plot(x2, y2, label='$\\tilde{g}(\\beta)$', color='green', linewidth=2)
    ax.plot([beta_bar, beta_bar], [-0.1 + g(beta_bar), g(x2[50]) - 0.1],
             label='$\\beta=\\bar{\\beta}$', linestyle='--')
    # draw the x-axis
    ax.plot([0, x2[50]], [0, 0], linestyle='--', color='black')
    ax.text(beta_star, 0, '$\\beta^*$', fontsize=16)
    plt.xlabel('$\\beta$', fontsize=12)
    plt.xlim(0, end_point + 0.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig('build/g-{0}-{1}-{2}.{3}'.format(int(a), int(b), k, fig_format))
    plt.show()

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
    for i in os.listdir('./'):
        if i.find(keyword_1) >= 0 and i.find(keyword_2) >= 0:
            Ls.append(i)
    return Ls

def draw_beta_phase_trans(date, pic_format='eps', theoretical=False):
    file_name_list = get_file_name_list('beta_trans', date + '.pickle')
    largest_n = 0
    line_width = 1
    if len(file_name_list) == 1:
        line_width = 4
    for file_name in file_name_list:
        f = open(file_name, 'rb')
        data = pickle.load(f)
        if data['n'] > largest_n:
            largest_n = data['n']
        if data.get('m') and data['m'] > 1:
            label_str = 'a=%.0f, b=%.0f, n=%d, k=%d, m=%d' % (data['a'], data['b'], data['n'], data['k'], data['m'])
        else:
            label_str = 'a=%.0f, b=%.0f, n=%d, k=%d' % (data['a'], data['b'], data['n'], data['k'])

        beta_list = data['beta_list']
        acc_list = data['acc_list']
        if len(acc_list) == 1:
            continue
        beta_star_empirical = compute_empirical_beta(acc_list, beta_list, data['k'])
        plt.plot(beta_list, acc_list, label=label_str, linewidth=line_width)
        if theoretical:
            plt.scatter([beta_star_empirical], [1.0 / data['k']], c='red', label='相变点')
            draw_theoretical_beta_phase_trans(largest_n, data['k'], data['a'], data['b'], beta_list[0], beta_list[-1])
        plt.scatter([beta_star_empirical], [1.0 / data['k']], c='red')

    L = plt.legend(edgecolor="black")
    plt.setp(L.texts, fontname='SimSun')
    L.get_frame().set_alpha(None)
    L.get_frame().set_facecolor((1, 1, 1, 0))
    plt.xlabel('$\\beta$', size='large')
    plt.ylabel('准\n确\n率', size='large', fontname='SimSun', rotation=0, labelpad=10)
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
    plt.plot(beta_list_1, acc_list_1, label='准确率上界', color='purple', linewidth=2)
    plt.plot(beta_list_2, acc_list_2, label='准确率下界', color='darkgreen', linewidth=2)
    plt.plot([beta_star, beta_star], [0, 1], label='相变分界线', color='red', linewidth=2, linestyle='dashed')

def animation_metropolis(n, k, a, b):
    G = sbm_graph(n, k, a, b)
    sibm = SIBM(G, k)
    # draw initial configuration with random labels
    skip_draw_count = 5
    N = 250
    energy_list = []
    # get initial energy
    energy_list.append(sibm._get_Hamiltonian())
    for i in range(N):
        for _ in range(skip_draw_count):
            sibm._metropolis_single()
        _animation_metropolis_inner(G, sibm.sigma, i)
        energy_list.append(sibm._get_Hamiltonian())
        # plot energy versus time(i)
        plt.plot(list(range(i + 2)), energy_list)
        plt.savefig('build/am/energy-%03d.png' % (i + 1))
        plt.clf()
    # create the final video by the following command
    # ffmpeg -r 20 -i %d.png -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" output.mp4

def show_animation():
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # check the number of images
    N = 250
    for i in range(1, N):
        im1 = mpimg.imread('build/am/%03d.png' % i)
        im1_rot = ndimage.rotate(im1, 90)
        imax1 = ax1.imshow(im1_rot)
        im2 = mpimg.imread('build/am/energy-%03d.png' % i)
        ax2.imshow(im2)
        plt.savefig('build/am/whole-%03d.png' % i)
        plt.clf()

def _animation_metropolis_inner(G, labels, n_index):
    ag = get_agraph(G, labels)
    ag.layout()
    ag.draw('build/am/%03d.png' % (n_index))

def draw_phase_transation(file_name, date, pic_format):
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
    plt.imshow(matrix, cmap='Greys_r', origin='lower', aspect=0.4, extent=[b_min, b_max, a_min, a_max])
    x = np.linspace(b_min, b_max)
    y = (np.sqrt(2) + np.sqrt(x)) ** 2
    plt.xlabel('b')
    plt.ylabel('a')
    plt.plot(x, y, color='red')
    plt.colorbar()
    fig_name = 'phase_trans-' + date + '.' + pic_format
    plt.savefig(os.path.join('build', fig_name), transparent=True)    
    plt.show()

if __name__ == '__main__':
    method_list = ['sdp', 'metropolis', 'asyn_fluid', 'bi', 'sdp2']
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['phase_transition',
        'compare', 'beta_transition', 'plot_g_function', 'animation_metropolis'], default='phase_transition')
    parser.add_argument('--method', choices=method_list, default='metropolis')
    parser.add_argument('--format', choices=['pdf', 'svg'], default='pdf')
    parser.add_argument('--a', type=float, default=16.0)
    parser.add_argument('--b', type=float, default=4.0)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--theoretical', const=True, default=False, nargs='?')
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()
    if args.action == 'phase_transition':
        file_name = args.method + '-transition-%s.pickle' % args.date
        draw_phase_transation(file_name, args.date, args.format)
    elif args.action == 'beta_transition':
        draw_beta_phase_trans(args.date, args.format, args.theoretical)
    elif args.action == 'plot_g_function':
        plot_g_function(args.a, args.b, args.k, fig_format=args.format)
    elif args.action == 'animation_metropolis':
        animation_metropolis(args.n, args.k, args.a, args.b)
    else:
        plot_alg_fix_b(method_list, args.date)
