import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

file_name = 'transition-2020-09-14.pickle'
with open(os.path.join('build', file_name), 'rb') as f:
    data = pickle.load(f)
a_list = data['a']
b_list = data['b']
acc_list = data['acc_list']
matrix = np.zeros([len(a_list), len(b_list)])
counter = 0
for i in range(len(a_list)):
    for j in range(len(b_list)):
        if a_list[i] <= b_list[j]:
            continue
        matrix[i, j] = acc_list[counter]
        counter += 1

plt.imshow(matrix, cmap='Greys')
plt.show()