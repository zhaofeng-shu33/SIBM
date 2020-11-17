# distutils: language = c++
from sibm_c cimport task_cpp
def task_cpp_wrapper(repeat, n, a, b, alpha, beta, inner_repeat, _N):
    return task_cpp(repeat, n, a, b, alpha, beta, inner_repeat, _N)
