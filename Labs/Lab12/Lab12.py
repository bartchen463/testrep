import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila


def create_rect(N, M):
    ''' this subroutine creates an ill-conditioned rectangular matrix'''
    a = np.linspace(1, 10, M)
    d = 10 ** (-a)

    D2 = np.zeros((N, M))
    for j in range(0, M):
        D2[j, j] = d[j]

    '''' create matrices needed to manufacture the low rank matrix'''
    A = np.random.rand(N, N)
    Q1, R = la.qr(A)
    test = np.matmul(Q1, R)
    A = np.random.rand(M, M)
    Q2, R = la.qr(A)
    test = np.matmul(Q2, R)

    B = np.matmul(Q1, D2)
    B = np.matmul(B, Q2)
    return B
''' create  matrix for testing different ways of solving a square 
linear system'''

'''' N = size of system'''
N = 100


def solveLU(A, b):
    return scila.lu_solve(scila.lu_factor(A), b)



for N in [100, 500, 1000, 2000, 3000, 4000, 5000]:
    b = np.random.rand(N, 1)
    A = np.random.rand(N, N)
    t0 = time.time()
    LU = scila.lu_factor(A)
    t1 = time.time()
    scila.lu_solve(LU, b)
    t2 = time.time()
    print(f'N={N}, LU Factor time: {t1 - t0}, total time: {t2 - t0}')
    t0 = time.time()
    scila.solve(A, b)
    t1 = time.time()
    print(f'N = {N}, Regular solve time: {t1 - t0}')


solveLU(A, b)
LU = scila.lu_factor(A)
scila.lu_solve(LU, b)


x = scila.solve(A, b)

test = np.matmul(A, x)
r = la.norm(test - b)

print(r)

''' Create an ill-conditioned rectangular matrix '''
N = 10
M = 5
A = create_rect(N, M)
b = np.random.rand(N, 1)


