import numpy as np
import numpy.linalg as la
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

N = 10
a = -1
b = 1
errs = []
for N in range(9,18):
    ''' Create interpolation nodes'''
    #xint = np.linspace(a, b, N + 1)
    jvals = np.arange(1, N + 1)
    xint = np.cos((2*jvals - 1)*np.pi/(2*N))
    #    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
    #    print('yint =',yint)

    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint, N)
    #    print('V = ',V)

    ''' Invert the Vandermonde matrix'''
    Vinv = inv(V)
    #    print('Vinv = ' , Vinv)

    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    coef = Vinv @ yint

    #    print('coef = ', coef)

    # No validate the code
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    yeval = eval_monomial(xeval, coef, N, Neval)

    # exact function
    yex = f(xeval)

    err = norm(yex - yeval)
    errs.append(err)
    print('N=', N, 'err = ', err)

plt.plot(xeval, yeval, label='Monomial Approximation')
plt.plot(xeval, yex, label='Exact')
plt.title(f"Monomial, N={N}")
plt.legend()
plt.show()


def eval_lagrange(xeval, xint, yint, N):
    lj = np.ones(N + 1)

    for count in range(N + 1):
        for jj in range(N + 1):
            if (jj != count):
                lj[count] = lj[count] * (xeval - xint[jj]) / (xint[count] - xint[jj])

    yeval = 0.

    for jj in range(N + 1):
        yeval = yeval + yint[jj] * lj[jj]

    return (yeval)


''' create divided difference matrix'''


def dividedDiffTable(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                       (x[j] - x[i + j]));
    return y;


def evalDDpoly(xval, xint, y, N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N + 1)

    ptmp[0] = 1.
    for j in range(N):
        ptmp[j + 1] = ptmp[j] * (xval - xint[j])

    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N + 1):
        yeval = yeval + y[0][j] * ptmp[j]

    return yeval


N = 10
errs_l = []
errs_dd = []
for N in range(9, 19):
    ''' interval'''
    a = -1
    b = 1

    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a, b, N + 1)

    ''' create interpolation data'''
    yint = f(xint)

    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    yeval_l = np.zeros(Neval + 1)
    yeval_dd = np.zeros(Neval + 1)

    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros((N + 1, N + 1))

    for j in range(N + 1):
        y[j][0] = yint[j]

    y = dividedDiffTable(xint, y, N + 1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval + 1):
        yeval_l[kk] = eval_lagrange(xeval[kk], xint, yint, N)
        yeval_dd[kk] = evalDDpoly(xeval[kk], xint, y, N)

    ''' create vector with exact values'''
    fex = f(xeval)

    # plt.figure()
    # plt.plot(xeval, fex, 'ro-')
    # plt.plot(xeval, yeval_l, 'bs--')
    # plt.plot(xeval, yeval_dd, 'c.--')
    # plt.legend()

    # plt.figure()
    err_l = norm(yeval_l - fex)
    err_dd = norm(yeval_dd - fex)
    print("N=", N, "Lagrange error=", err_l, "DD Error=", err_dd)

plt.plot(xeval, yeval_l, label='Lagrange Approximation')
plt.plot(xeval, yeval_dd, label='Divided Difference')
plt.plot(xeval, fex, label='Exact')
plt.title(f'N={N}')
plt.legend()
plt.show()

plt.semilogy(xeval, err_l, 'ro--', label='lagrange')
plt.semilogy(xeval, err_dd, 'bs--', label='Newton DD')
plt.legend()
plt.show()




