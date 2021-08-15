import networkx as nx
import numpy as np

def sbm_graph(n, k, a, b):
    if n % k != 0:
        raise ValueError('n %k != 0')
    elif a <= b:
        raise ValueError('a <= b')
    sizes = [int(n/k) for _ in range(k)]
    _p = np.log(n) * a / n
    _q = np.log(n) * b / n
    if _p > 1 or _q > 1:
        raise ValueError('')
    p = np.diag(np.ones(k) * (_p - _q)) + _q * np.ones([k, k])
    return nx.generators.community.stochastic_block_model(sizes, p)