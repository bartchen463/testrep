import numpy as np
h = 0.01 * 2. ** (-np.arange(0,10))


fdiff = (np.cos(np.pi/2 + h) - np.cos(np.pi/2))/h
cdiff = (np.cos(np.pi/2 + h) - np.cos(np.pi/2 - h))/(2*h)

fdiff + 1
cdiff + 1



import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm

import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm

def evalF(x):
    F = np.zeros(2)

    F[0] = 4*x[0]**2 + x[1]**2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0] - x[1])
    return F


def evalJ(x):
    J = np.array([[8*x[0], 2*x[1]], [1-np.cos(x[0] - x[1]), 1+np.cos(x[0] - x[1])]])
    return J

def LazyNewton(x0, tol, Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if np.linalg.norm(x1)>np.linalg.norm(x0):
            J = evalJ(x0)
            Jinv = inv(J)
        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]

x0 = np.array([1,0])

Nmax = 100
tol = 1e-10
t = time.time()
for j in range(20):
    [xstar, ier, its] = LazyNewton(x0, tol, Nmax)
elapsed = time.time() - t
print(xstar)
print('Lazy Newton: the error message reads:', ier)
print('Lazy Newton: took this many seconds:', elapsed / 20)
print('Lazy Newton: number of iterations is:', its)



def Newton(x0, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        hx = 1e-7 * np.abs(x0)[0]
        hy = 1e-7 * np.abs(x0)[1] + 0.00001
        fx = evalF(x0)[0]
        fy = evalF(x0)[1]
        fxh = (evalF(x0 + np.array([hx,0])) - fx)/hx
        fyh = (evalF(x0 + np.array([0,hy])) - fy)/hy
        J = np.array([[fxh[0], fxh[1]], [fyh[0], fyh[1]]])
        Jinv = inv(J)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


x0 = np.array([1,0])

Nmax = 100
tol = 1e-10

t = time.time()
for j in range(50):
    [xstar, ier, its] = Newton(x0, tol, Nmax)
elapsed = time.time() - t
print(xstar)
print('Newton: the error message reads:', ier)
print('Newton: took this many seconds:', elapsed / 50)
print('Netwon: number of iterations is:', its)





def Broyden(x0, tol, Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''

    '''Sherman-Morrison 
    (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''

    '''In Broyden 
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''

    '''initialize with 1 newton step'''

    A0 = evalJ(x0)

    v = evalF(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0 + s
    for its in range(Nmax):
        '''(save v from previous step)'''
        w = v
        ''' create new v'''
        v = evalF(xk)
        '''y_k = F(xk)-F(xk-1)'''
        y = v - w;
        '''-A_{k-1}^{-1}y_k'''
        z = -A.dot(y)
        ''' p = s_k^tA_{k-1}^{-1}y_k'''
        p = -np.dot(s, z)
        u = np.dot(s, A)
        ''' A = A_k^{-1} via Morrison formula'''
        tmp = s + z
        tmp2 = np.outer(tmp, u)
        A = A + 1. / p * tmp2
        ''' -A_k^{-1}F(x_k)'''
        s = -A.dot(v)
        xk = xk + s
        if (norm(s) < tol):
            alpha = xk
            ier = 0
            return [alpha, ier, its]
    alpha = xk
    ier = 1
    return [alpha, ier, its]


x0 = np.array([0.1, 0.1, -0.1])

Nmax = 100
tol = 1e-10

t = time.time()
for j in range(50):
    [xstar, ier, its] = Newton(x0, tol, Nmax)
elapsed = time.time() - t
print(xstar)
print('Newton: the error message reads:', ier)
print('Newton: took this many seconds:', elapsed / 50)
print('Netwon: number of iterations is:', its)

t = time.time()




t = time.time()
for j in range(50):
    [xstar, ier, its] = Newton(x0, tol, Nmax)
elapsed = time.time() - t
print(xstar)
print('Newton: the error message reads:', ier)
print('Newton: took this many seconds:', elapsed / 50)
print('Netwon: number of iterations is:', its)



t = time.time()
for j in range(20):
    [xstar, ier, its] = Broyden(x0, tol, Nmax)
elapsed = time.time() - t
print(xstar)
print('Broyden: the error message reads:', ier)
print('Broyden: took this many seconds:', elapsed / 20)
print('Broyden: number of iterations is:', its)


