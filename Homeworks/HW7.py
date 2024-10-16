import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt

def eval_monomial(xeval, coef, N, Neval):
    N=N-1
    yeval = coef[0] * np.ones(Neval + 1)

    #    print('yeval = ', yeval)

    for j in range(1, N + 1):
        for i in range(Neval + 1):
            #        print('yeval[i] = ', yeval[i])
            #        print('a[j] = ', a[j])
            #        print('i = ', i)
            #        print('xeval[i] = ', xeval[i])
            yeval[i] = yeval[i] + coef[j] * xeval[i] ** j

    return yeval


def Vandermonde(xint, N):
    N = N-1
    V = np.zeros((N + 1, N + 1))

    ''' fill the first column'''
    for j in range(N + 1):
        V[j][0] = 1.0

    for i in range(1, N + 1):
        for j in range(N + 1):
            V[j][i] = xint[j] ** i

    return V


f = lambda x: 1 / (1 + (10 * x)**2)

N = 3
for N in [3, 13, 21]:
    a = -1
    b = 1
    xint = np.linspace(a, b, N)
    yint = f(xint)
    V = Vandermonde(xint, N)
    Vinv = inv(V)
    coef = Vinv @ yint
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    yeval = eval_monomial(xeval, coef, N, Neval)
    yex = f(xeval)
    err = np.abs(yex - yeval)
    plt.plot(xeval, yeval, label='Monomial Approximation')
    plt.plot(xeval, yex, label='Exact')
    plt.scatter(xint, yint, label='Interpolation Nodes', marker='o', color='r')
    plt.title(f"Monomial, N={N}")
    plt.legend(loc='best')
    plt.show()

for N in [3, 13, 21]:
    a = -1
    b = 1
    xint = np.linspace(a, b, N)
    yint = f(xint)
    w = np.zeros(N)
    for j in range(N):
        xj = xint[j]
        wj = 1.0
        for i in range(N):
            if i != j:
                wj *= xj - xint[i]
        w[j] = 1/wj
    xeval = np.array([x for x in np.linspace(a, b, 1001) if x not in xint])
    yex = f(xeval)
    yeval = np.zeros(len(xeval))
    for i in range(len(xeval)):
        num = 0
        denom = 0
        for j in range(N):
            z = w[j]/(xeval[i] - xint[j])
            num += z * yint[j]
            denom += z
        yeval[i] = num / denom
    err = np.abs(yex - yeval)
    plt.plot(xeval, yeval, label='Barycentric Lagrange Approximation')
    plt.plot(xeval, yex, label='Exact')
    plt.scatter(xint, yint, label='Interpolation Nodes', marker='o', color='r')
    plt.title(f"Barycentric Lagrange, N={N}")
    plt.legend(loc='best')
    plt.show()


for N in [3, 13, 21, 35]:
    a = -1
    b = 1
    jvals = np.arange(1, N + 1)
    xint = np.cos((2 * jvals - 1) * np.pi / (2 * N))
    yint = f(xint)
    V = Vandermonde(xint, N)
    Vinv = inv(V)
    coef = Vinv @ yint
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    yeval = eval_monomial(xeval, coef, N, Neval)
    yex = f(xeval)
    err = np.abs(yex - yeval)
    plt.plot(xeval, yeval, label='Monomial Approximation')
    plt.plot(xeval, yex, label='Exact')
    plt.scatter(xint, yint, label='Interpolation Nodes', marker='o', color='r')
    plt.title(f"Monomial, Chebychev, N={N}")
    plt.legend(loc='best')
    plt.show()

for N in [3, 13, 21, 35]:
    a = -1
    b = 1
    jvals = np.arange(1, N + 1)
    xint = np.cos((2 * jvals - 1) * np.pi / (2 * N))
    yint = f(xint)
    w = np.zeros(N)
    for j in range(N):
        xj = xint[j]
        wj = 1.0
        for i in range(N):
            if i != j:
                wj *= xj - xint[i]
        w[j] = 1/wj
    xeval = np.array([x for x in np.linspace(a, b, 1001) if x not in xint])
    yex = f(xeval)
    yeval = np.zeros(len(xeval))
    for i in range(len(xeval)):
        num = 0
        denom = 0
        for j in range(N):
            z = w[j]/(xeval[i] - xint[j])
            num += z * yint[j]
            denom += z
        yeval[i] = num / denom
    err = np.abs(yex - yeval)
    plt.plot(xeval, yeval, label='Barycentric Lagrange Approximation')
    plt.plot(xeval, yex, label='Exact')
    plt.scatter(xint, yint, label='Interpolation Nodes', marker='o', color='r')
    plt.title(f"Barycentric Lagrange, Chebychev, N={N}")
    plt.legend(loc='best')
    plt.show()



