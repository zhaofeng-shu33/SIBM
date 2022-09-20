from scipy.optimize import minimize, Bounds
import numpy as np

def g(a, b, e):
    return a + b - np.sqrt(e ** 2 + 4 * a * b) + e * np.log((e + np.sqrt(e ** 2 + 4 * a * b))/ (2 * b))
p0 = 0.2
p1 = 0.8
gamma = 1
kappa = 0.1
a = 16
b = 4
tilde_gamma = 2 / (kappa + 1) * gamma

def target_function(x):
    D_1 = x * np.log(x/p0) + (1-x) * np.log((1-x)/(1-p0))
    D_2 = x * np.log(x/p1) + (1-x) * np.log((1-x)/(1-p1))
    eps = tilde_gamma * (D_2 - D_1)/ np.log(a/b)
    return gamma*D_1 + 0.5*g(a,b,2*eps)

bounds = Bounds(1e-3, 1-1e-3)
res = minimize(target_function, x0=0.3, bounds=bounds)
print(res)