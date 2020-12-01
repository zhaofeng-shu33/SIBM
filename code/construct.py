import argparse
import numpy as np
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--action', choices=['eig', 'side'], default='side')
    args = parser.parse_args()
    if args.action == 'side':
        SDP_side_info()
    if args.action == 'eig':
        Eig_verification(args.k)