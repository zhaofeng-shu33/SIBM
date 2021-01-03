import argparse
import numpy as np

from multiprocessing import Queue
from multiprocessing import Process

from sbm import sbm_graph
from sdp import sdp2
from sibm_experiment import exact_compare_k

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
    print(np.linalg.eig(B)[0])
    c = n / 2 - beta * np.log(n)
    L3 = np.array([[n2 *(D1 + D2), -n2 * D1, n2 * D2], [-D1, D1 + c, c],[D2, c, D2 + c]])
    print(np.linalg.eig(L3)[0])
    a2 = -(D1 + D2 + 2 * c + n / 2 * (D1 + D2))
    a3 = (1 + n) * (D1 * D2 + D1 * c + D2 * c)
    print(np.roots([1, a2, a3]))
    print(d  - a + D1)
    print(d  - a + D2)

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
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1, help='number of times to generate the SBM graph')
    parser.add_argument('--action', choices=['eig', 'side', 'sdp'], default='side')
    args = parser.parse_args()
    if args.action == 'side':
        SDP_side_info()
    if args.action == 'eig':
        Eig_verification(args.k)
    if args.action == 'sdp':
        print(get_acc(args.repeat, args.n, args.a, args.b, args.kappa, args.thread_num))
