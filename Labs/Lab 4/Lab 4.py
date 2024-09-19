import numpy as np

def fixedpt(f, x0, tol, Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    x = np.zeros((Nmax, 1))  # Store iterates
    count = 0
    while (count < Nmax):
        x1 = f(x0)
        x[count] = x1  # Store the current value
        count += 1
        if abs(x1 - x0) < tol:
            xstar = x1
            x[count] = xstar
            ier = 0  # ier = 0 indicates convergence
            return x[:count+1], xstar, ier  # Return up to the converged value
        x0 = x1

    xstar = x1
    ier = 1  # ier = 1 indicates no convergence after Nmax iterations
    return x, xstar, ier


f1 = lambda x: -np.sin(2 *x) + (5 * x) / 4 - 3 / 4


Nmax = 1000
tol = 1e-10

# test f1 '''
x0 = 2.5
[x, xstar, ier] = fixedpt(f1, x0, tol, Nmax)
print('the approximate fixed point is:', xstar)
print('f1(xstar):', f1(xstar))
print('Error message reads:', ier)