import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import quad
def comp_trap(a, b, f, N):
    nodes = np.linspace(a, b, N)
    weights = np.full(N, 2).astype(float)
    weights[0] = 1
    weights[-1] = 1
    h = (b-a)/(N-1)
    weights *= h/2
    return np.dot(f(nodes), weights)

def comp_simp(a, b, f, N):
    if N%2 == 0:
        return print("N must be odd")
    nodes = np.linspace(a, b, N)
    weights = np.zeros(N)
    for i in range(N):
        if i == 0 or i == N-1:
            weights[i] = 1
        elif i%2 == 1:
            weights[i] = 4
        else:
            weights[i] = 2
    h = ((b-a)/(N-1)) * 2
    weights *= h/6
    return np.dot(f(nodes), weights)

def f(x):
    return 1/(1 + x**2)

comp_trap(-5,5, f, 200)
comp_simp(-5, 5, f, 201)

quad6, err6, info6 = scipy.integrate.quad(f, -5, 5, epsabs=1e-6, full_output=True)
quad4, err4, info4 = scipy.integrate.quad(f, -5, 5, epsabs=1e-4, full_output=True)

np.abs(quad6 - comp_trap(-5, 5, f, 1292))
np.abs(quad6 - comp_simp(-5, 5, f, 193))
np.abs(quad4 - comp_trap(-5, 5, f, 1292))
np.abs(quad4 - comp_simp(-5, 5, f, 193))

info6['neval']
info4['neval']

def g(x):
    return x * np.cos(1/x)

comp_simp(1e-8, 1, g, 5)