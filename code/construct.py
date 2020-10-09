import numpy as np
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
