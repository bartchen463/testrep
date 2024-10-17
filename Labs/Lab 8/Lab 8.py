import numpy as np
import math
import matplotlib.pyplot as plt
def line(x0, x1, fx0, fx1, a):
    m = (fx1 - fx0) / (x1 - x0)
    return (a - x0)*m + fx0

line(0,5, 2, 4, 6)


f = lambda x: 1 / (1 + (10*x)**2)
a = -1
b = 1


def eval_lin_spline(f, a, b, Nint, Neval):
    xint = np.linspace(a, b, Nint)
    yint = f(xint)
    xeval = np.linspace(a, b, Neval)
    yex = f(xeval)
    yeval = np.zeros(Neval)
    for j in range(len(xeval)):
        x = xeval[j]
        i = 0
        while(x>xint[i]):
            i += 1
        yeval[j] = line(xint[i-1], xint[i], yint[i-1], yint[i], x)
    plt.plot(xeval, yex, label = 'exact')
    plt.plot(xeval, yeval, label = 'linear spline approximation')
    plt.scatter(xint, yint, label = "Interpolation Nodes")
    plt.legend(loc = 'best')
    plt.title(f'N = {Nint} Linear Spline')
    plt.show()

eval_lin_spline(f, a, b, 60, 1000)

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm


# def create_natural_spline(yint, xint, N):
#     #    create the right  hand side for the linear system
#     b = np.zeros(N + 1)
#     #  vector values
#     h = np.zeros(N + 1)
#     # h[0] = xint[i] - xint[i - 1]
#     for i in range(N):
#         h[i] = xint[i + 1] - xint[i]
#         b[i] = (yint[i + 1] - yint[i]) / h[i] - (yint[i] - yint[i - 1]) / h[i - 1]
#
#     #  create the matrix A so you can solve for the M values
#     A = np.zeros((N + 1, N + 1))
#     for i in range(N+1):
#         for j in range(N+1):
#             if i == 0 and j == 0 or i == N or j == N:
#                 A[i, j] = 1
#             if i == j+1:
#                 A[i, j] = h[j] / 6
#             if i == j and i!=0:
#                 A[i, j] = h[i] + h[i-1] / 3
#             if j == i + 2:
#                 A[i,j] = h[i] / 6
#
#
#     #  Invert A
#     Ainv = np.linalg.inv(A)
#
#     # solver for M
#     M = Ainv @ b
#
#     #  Create the linear coefficients
#     C = np.zeros(N)
#     D = np.zeros(N)
#     for j in range(N):
#         C[j] =  # find the C coefficients
#         D[j] =  # find the D coefficients
#     return (M, C, D)
#
#
# def eval_local_spline(xeval, xi, xip, Mi, Mip, C, D):
#     # Evaluates the local spline as defined in class
#     # xip = x_{i+1}; xi = x_i
#     # Mip = M_{i+1}; Mi = M_i
#
#     hi = xip - xi
#
#     yeval =
#     return yeval
#
#
# def eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D):
#     yeval = np.zeros(Neval + 1)
#
#     for j in range(Nint):
#         '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
#         '''let ind denote the indices in the intervals'''
#         atmp = xint[j]
#         btmp = xint[j + 1]
#
#         #   find indices of values of xeval in the interval
#         ind = np.where((xeval >= atmp) & (xeval <= btmp))
#         xloc = xeval[ind]
#
#         # evaluate the spline
#         yloc = eval_local_spline(xloc, atmp, btmp, M[j], M[j + 1], C[j], D[j])
#         #   copy into yeval
#         yeval[ind] = yloc
#
#     return (yeval)
#
#
#
# f = lambda x: np.exp(x)
# a = 0
# b = 1
#
# ''' number of intervals'''
# Nint = 3
# xint = np.linspace(a, b, Nint + 1)
# yint = f(xint)
#
# ''' create points you want to evaluate at'''
# Neval = 100
# xeval = np.linspace(xint[0], xint[Nint], Neval + 1)
#
# #   Create the coefficients for the natural spline
# (M, C, D) = create_natural_spline(yint, xint, Nint)
#
# #  evaluate the cubic spline
# yeval = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)
#
# ''' evaluate f at the evaluation points'''
# fex = f(xeval)
#
# nerr = norm(fex - yeval)
# print('nerr = ', nerr)
#
# plt.figure()
# plt.plot(xeval, fex, 'ro-', label='exact function')
# plt.plot(xeval, yeval, 'bs--', label='natural spline')
# plt.legend
# plt.show()
#
# err = abs(yeval - fex)
# plt.figure()
# plt.semilogy(xeval, err, 'ro--', label='absolute error')
# plt.legend()
# plt.show()
#



