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



def compute_order(x, xstar):
    diff1 = np.abs(x[1:] - xstar)
    diff2 = np.abs(x[0:-1] - xstar)

    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)
    _lambda = np.exp(fit[0])
    alpha = fit[0]
    print(f"Lambda is {_lambda}")
    print(f"Alpha is {alpha}")
    return fit


f1 = lambda x: np.sqrt(10/(x+4))


Nmax = 1000
tol = 1e-10

# test f1 '''
x0 = 1.5
[x, xstar, ier] = fixedpt(f1, x0, tol, Nmax)
print('the approximate fixed point is:', xstar)
print('f1(xstar):', f1(xstar))
print('Error message reads:', ier)

compute_order(x, 1.3652300134140976)

def Aitken(x):
    phats = [x[i] - ((x[i+1] - x[i]) ** 2)/(x[i+2] - 2*x[i+1] + x[i]) for i in range(len(x) - 2)]
    return np.array(phats)

(Aitken(x) - 1.3652300134140976)/(x - 1.3652300134140976)[:-2]

compute_order(Aitken(x), 1.3652300134140976)