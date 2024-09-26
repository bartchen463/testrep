import numpy as np
import scipy
from scipy import special
import matplotlib.pyplot as plt

x = np.linspace(0,2,500)
def f(x):
    return 35 * scipy.special.erf(x / (2 * np.sqrt(0.715392))) - 15
y = f(x)
plt.plot(x, y)
plt.title(f'$f(x)$ = Temperature at depth x on day 60')
plt.axhline(y=0, color='k')
plt.xlabel('x (meters)')
plt.ylabel('f(x) (Degrees Celsius)')
plt.show()

def bisection(f, a, b, tol):
    #    Inputs:
    #     f,a,b       - function and endpoints of initial interval
    #      tol  - bisection stops when interval length < tol

    #    Returns:
    #      astar - approximation of root
    #      ier   - error message
    #            - ier = 1 => Failed
    #            - ier = 0 == success

    #     first verify there is a root we can find in the interval

    fa = f(a)
    fb = f(b);
    if (fa * fb > 0):
        ier = 1
        astar = a
        return [astar, ier]

    #   verify end points are not a root
    if (fa == 0):
        astar = a
        ier = 0
        return [astar, ier]

    if (fb == 0):
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    d = 0.5 * (a + b)
    while (abs(d - a) > tol):
        fd = f(d)
        print(count)
        if (fd == 0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa * fd < 0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        count = count + 1
    #      print('abs(d-a) = ', abs(d-a))

    astar = d
    ier = 0
    return [astar, ier]

bisection(f, 0, 2, 1e-13)

def newton(f,fp,p0,tol,Nmax):
    """
    Newton iteration.
    Inputs:
    f,fp - function and derivative
    p0 - initial guess for root
    tol - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
    Returns:
    p - an array of the iterates
    pstar - the last iterate
    info - success message
    - 0 if we met tol
    - 1 if we hit Nmax iterations (fail)
    """
    p = [p0]
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p.append(p1)
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [np.array(p),pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [np.array(p),pstar,info,it]

fp = lambda x: (35)/(np.sqrt(np.pi*0.715392)) * np.exp(-x**2/(4 * 0.715392))
p0 = 2
Nmax = 100
tol = 1e-13
(p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
print('the approximate root is', '%16.16e' % pstar)
print('the error message reads:', '%d' % info)
print('Number of iterations:', '%d' % it)

def secant(f, x0, x1, Nmax, tol):
    pvals = [x0, x1]
    if np.abs(f(x0) - f(x1)) == 0:
        ier = 1
        p = x1
        return [np.array(pvals), p, ier]
    for i in range(Nmax):
        x2 = x1 - (f(x1)*(x1 - x0))/(f(x1) - f(x0))
        pvals.append(x2)
        if np.abs(x1 - x2) < tol:
            p = x2
            ier = 0
            return [np.array(pvals), p, ier]
        x0 = x1
        x1 = x2
        if np.abs(f(x1) - f(x0)) == 0:
            p = x2
            ier = 1
            return [np.array(pvals), p, ier]
    p = x2
    ier = 1
    return [np.array(pvals), p, ier]

def f1(x):
    return x**6 - x - 1
def f1p(x):
    return 6*x**5 - 1

Nmax = 100
tol = 1e-13
p0 = 2
(p, pstar, info, it) = newton(f1,f1p,p0,tol,Nmax)

(pvals, p1, ier) = secant(f1,2,1,Nmax,tol)

p - 1.134724138402
pvals - 1.134724138402

xk1 = np.abs((p - 1.134724138402)[1:])
xk = np.abs((p - 1.134724138402)[:-1])
plt.loglog(xk, xk1)
plt.title('Log Error in $x_{n+1}$ vs $x_n$, Newton method')
plt.xlabel('Error $x_n$')
plt.ylabel('Error $x_{n+1}$')
plt.show()

xk1 = np.abs((pvals - 1.134724138402)[1:])
xk = np.abs((pvals - 1.134724138402)[:-1])
plt.loglog(xk, xk1)
plt.title('Log Error in $x_{n+1}$ vs $x_n$, Secant method')
plt.xlabel('Error $x_n$')
plt.ylabel('Error $x_{n+1}$')
plt.show()