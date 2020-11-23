# distutils: language = c++
from sibm_c cimport task_cpp
def task_cpp_wrapper(repeat, n, k, a, b, alpha, beta, num_of_sibm_samples, m, _N):
    return task_cpp(repeat, n, k, a, b, alpha, beta, num_of_sibm_samples, m, _N)
