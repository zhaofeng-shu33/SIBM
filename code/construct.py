import argparse
import numpy as np
from scipy.optimize import minimize, Bounds
from matplotlib import pyplot as plt

from multiprocessing import Queue
from multiprocessing import Process

from sbm import sbm_graph
from sdp import sdp2
from sibm_experiment import exact_compare_k
from sdp import construct_B

def g(a, b, e):
    return a + b - np.sqrt(e ** 2 + 4 * a * b) + e * np.log((e + np.sqrt(e ** 2 + 4 * a * b))/ (2 * b))

def target_function(x, c1, c2, p0, p1, gamma):
    return np.power(c1, 1 - x) * np.power(c2, x) + np.power(c2, 1 - x) * np.power(c1, x) + \
        gamma * np.log(np.power(p0, 1 - x) * np.power(p1, x) + np.power(1 - p0, 1 - x) * np.power(1 - p1, x))

def sample_ab_abbe_recover_or_not(p0, p1, gamma, num=100, draw_eps=False):    
    b_list = np.linspace(0.1, 10, num=num)
    b_min, b_max = b_list[0], b_list[num - 1]
    a_list = np.linspace(0.1, 20, num=num)
    a_min, a_max = a_list[0], a_list[num - 1]

    # plt.imshow(Z, cmap='Greys_r', origin='lower', extent=[b_min, b_max, a_min, a_max])
    x = np.linspace(b_min, b_max)
    y = (np.sqrt(2) + np.sqrt(x)) ** 2
    plt.xlabel('b')
    plt.ylabel('a')
    plt.plot(x, y, color='blue', label='SBM only', linewidth=3)
    D12 = -2 * np.log(np.sqrt(p0 * p1) + np.sqrt((1 - p0) * (1 - p1)))
    y = (np.sqrt(2 - gamma * D12) + np.sqrt(x)) ** 2
    plt.plot(x, y, color='red', label='SBMSI with equal community size', linewidth=3)
    abbe_a_list = []
    for b in x:
        abbe_a_list.append(bisection_a(b, p0, p1, gamma))
    plt.plot(x, abbe_a_list, color='green', label='SBMSI with equal community probability', linewidth=3)
    if draw_eps:
        _title = 'comparison'
        suffix = '.eps'
    else:
        _title = 'p0={0},p1={1},gamma={2}'.format(p0, p1, gamma)
        suffix = '.png'
        plt.title(_title)
    plt.legend()
    plt.savefig('build/' + _title + suffix)
    plt.show()

def abbe_recover_or_not(a, b, p0, p1, gamma=1.0):
    bounds = Bounds(0, 1)
    c1 = a / 2
    c2 = b / 2
    res = minimize(target_function, x0=[0.3], args=(c1 / 2, c2 / 2, p0, p1, gamma), bounds=bounds)
    x = res.x[0]
    p_lambda = np.power(p0, 1 - x) * np.power(p1, x) / (np.power(p0, 1 - x) * np.power(p1, x) + \
        np.power(1 - p0, 1 - x) * np.power(1 - p1, x))
    D_KL = p_lambda * np.log(p_lambda / p0) + (1 - p_lambda) * np.log((1 - p_lambda) / (1 - p0))
    I_plus = gamma * D_KL - np.power(c1, 1 - x) * np.power(c2, x) - np.power(c2, 1 - x) * np.power(c1, x)
    I_plus += c1 + c2 + x * (np.power(c1, 1 - x) * np.power(c2, x) - \
        np.power(c2, 1 - x) * np.power(c1, x)) * np.log(c2 / c1)
    return I_plus

def bisection_a(b, p0, p1, gamma, tol=1e-2):
    a_left = b + tol
    a_right = (np.sqrt(b) + np.sqrt(2)) ** 2
    is_continue = True
    while is_continue:
        a_middle = (a_left + a_right) / 2
        if abbe_recover_or_not(a_middle, b, p0, p1, gamma) > 1:
            a_right = a_middle
        else:
            a_left = a_middle
        if np.abs(a_right - a_left) < tol:
            is_continue = False
    return a_middle

def verify_improvement():
    b = 4 * np.random.random() + np.random.random()
    a = (np.sqrt(2) + np.random.random() + np.sqrt(b)) ** 2
    p0 = np.random.random() # 0.4
    p1 = np.random.random() #0.7
    gamma = 1.0 # 5.0
    D2 = p1 * np.log(p1 / p0) + (1 - p1) * np.log( (1 - p1) / (1 - p0))
    D1 = p0 * np.log(p0 / p1) + (1 - p0) * np.log( (1 - p0) / (1 - p1))
    z = np.linspace(0.01, 0.99)
    epsilon = gamma / np.log(a / b) * ((D2 - D1) / 2 + z * np.log(p0 / p1) + (1 - z) * np.log((1 - p0) / (1 - p1)))
    y = gamma * (z * np.log(z / p0) + (1 - z) * np.log((1 - z) / (1 - p0))) + 0.5 * g(a, b, 2 * epsilon)
    # D12 = -1 * np.log(np.sqrt(p0 * p1) + np.sqrt((1 - p0) * (1 - p1)))
    hat_y = y - 0.5 * (np.sqrt(a) - np.sqrt(b)) ** 2  #- D12
    print(a, b, p0, p1, np.min(hat_y))
    # plt.plot(z, hat_y)
    # plt.show()

def SDP_random_matrix():
    n = 500
    n2 = int(n / 2)
    a = 11.66
    b = 4
    p0 = 0.4
    p1 = 0.7
    gamma = 6.0 # 5.0
    m = int(gamma * np.log(n))
    # gamma = m / np.log(n) * 1.0
    h_vector = np.zeros([n])

    tmp_1 = np.random.binomial(m, p0, size=n2)
    tmp_1 = tmp_1 * np.log(p0 / p1) + (m - tmp_1) * np.log((1 - p0)/ (1 - p1))
    h_vector[:n2] = tmp_1 / np.log (a / b)
    tmp_1 = np.random.binomial(m, p1, size=n2)
    tmp_1 = tmp_1 * np.log(p0 / p1) + (m - tmp_1) * np.log((1 - p0)/ (1 - p1))
    h_vector[n2:] = tmp_1 / np.log (a / b)

    G = sbm_graph(n, 2, a, b)
    B_matrix = construct_B(G)
    g = np.ones([n])
    g[n2:] = -1.0
    tmp_m = np.kron(h_vector, g) + np.kron(B_matrix @ g, g)
    block_matrix = np.diag(np.diag(tmp_m.reshape([n, n]))) - B_matrix
    values, _ = np.linalg.eig(block_matrix)
    tmp_2 = np.diag(np.kron(B_matrix @ g, g).reshape([n, n]))
    values_2, _ = np.linalg.eig(np.diag(tmp_2) - B_matrix)
    print(np.min(values), np.min(values_2))
    print(np.all(h_vector * g > 0))
    return np.min(values) > 0

def SDP_side_info():
    n = 128
    n2 = int(n / 2)
    alpha = 16
    beta = 4
    D1 = 2
    D2 = 3
    a = -2 * alpha * np.log(n) / n + 1
    b = -2 * beta * np.log(n) / n + 1
    d = (alpha - beta) * np.log(n) - 2 * alpha * np.log(n) / n + 1
    B = np.zeros([n + 1, n + 1])
    B[0, 0] = n2 * (D1 + D2)
    for i in range(n2):
        B[i + 1, 0] = -D1 # lower triangle
        B[i + 1, i + 1] = D1 + d
        B[i + 1 + n2, 0] = D2
        B[i + 1 + n2, i + 1 + n2] = D2 + d
    for i in range(2, n2 + 1):
        for j in range(1, i):
            B[i, j] = a
    for i in range(n2 + 2, n + 1):
        for j in range(n2 + 1, i):
            B[i, j] = a
    for i in range(n2 + 1, n + 1):
        for j in range(1, n2 + 1):
            B[i, j] = b
    for i in range(n):
        for j in range(i+1, n + 1):
            B[i, j] = B[j, i]
    print(np.linalg.eig(B[1:,1:])[0])
    c = n / 2 - beta * np.log(n)
    # L3 = np.array([[n2 *(D1 + D2), -n2 * D1, n2 * D2], [-D1, D1 + c, c],[D2, c, D2 + c]])
    #print(np.linalg.eig(L3)[0])
    d_prime = D1 + d - a
    c_prime = D2 + d - a
    a2 = -(d_prime + c_prime + n * a)
    a3 = d_prime * c_prime + n2 * a * (d_prime + c_prime) + n2 ** 2 * (a ** 2 - b ** 2)
    print(np.roots([1, a2, a3]))
    print(d + D1 - a)
    print(d + D2 - a)

def Eig_verification(k=3):
    n = 10
    d = 4
    a = 2
    b = 1
    B = np.ones([k * n, k * n]) * b
    for i in range(k * n):
        B[i, i] = d
    for i in range(k):
        for j_1 in range(n):
            for j_2 in range(n):
                if j_1 == j_2:
                    continue
                B[i * n + j_1, i * n + j_2] = a
    val, vec = np.linalg.eig(B)
    print(val)

def get_acc_sdp2(repeat, n, a, b, kappa, queue=None):
    acc = 0
    for _ in range(repeat):
        graph = sbm_graph(n, 2, a, b)
        results = sdp2(graph, kappa)
        acc += float(exact_compare_k(results, 2))
    acc /= repeat
    if queue:
        queue.put(acc)
    return acc

def get_acc(repeat, n, a, b, kappa, thread_num):
    if thread_num == 1:
        return get_acc_sdp2(repeat, n, a, b, kappa)
    q = Queue()
    process_list = []
    assert(repeat % thread_num == 0)
    inner_repeat = repeat // thread_num
    for _ in range(thread_num):
        t = Process(target=get_acc_sdp2,
                    args=(inner_repeat, n, a, b, kappa, q))
        process_list.append(t)
        t.start()
    acc = 0
    for i in range(thread_num):
        process_list[i].join()
        acc += q.get()
    acc /= thread_num
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--n', type=int, default=120)
    parser.add_argument('--a', type=float, default=16)
    parser.add_argument('--b', type=float, default=4)
    parser.add_argument('--kappa', type=float, default=1)
    parser.add_argument('--p0', type=float, default=0.5)
    parser.add_argument('--p1', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--eps', type=bool, const=True, nargs='?', default=False)
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1, help='number of times to generate the SBM graph')
    parser.add_argument('--action', choices=['eig', 'side', 'sdp', 'random', 'improve', 'abbe_recover', 'abbe_recover_repeat'], default='side')
    args = parser.parse_args()
    if args.action == 'side':
        SDP_side_info()
    if args.action == 'eig':
        Eig_verification(args.k)
    if args.action == 'sdp':
        print(get_acc(args.repeat, args.n, args.a, args.b, args.kappa, args.thread_num))
    if args.action == 'random':
        SDP_random_matrix()
    if args.action == 'improve':
        verify_improvement()
    if args.action == 'abbe_recover':
        print(abbe_recover_or_not(args.a, args.b, 0.9, 0.1))
        # compare with 1
    if args.action == 'abbe_recover_repeat':
        sample_ab_abbe_recover_or_not(args.p0, args.p1, args.gamma, draw_eps=args.eps)