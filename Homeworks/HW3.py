import numpy as np
import matplotlib.pyplot as plt

# define routines
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


    # use routines
f = lambda x: 2*x - 1 - np.sin(x)
a = 0
b = np.pi
tol = 1e-8

[astar, ier] = bisection(f, a, b, tol)
print('the approximate root is', astar)
print('the error message reads:', ier)
print('f(astar) =', f(astar))

f = lambda x: (x - 5) ** 9
a = 4.82
b = 5.2
tol = 1e-4

[astar, ier] = bisection(f, a, b, tol)
print('the approximate root is', astar)
print('the error message reads:', ier)
print('f(astar) =', f(astar))


f = lambda x: x ** 9 - 45 * x ** 8 + 900 * x ** 7 - 10500 * x ** 6 + 78750 * x ** 5 - 393750 * x ** 4\
    + 1312500 * x ** 3 - 2812500 * x ** 2 + 3515625 * x - 1953125
a = 4.82
b = 5.2
tol = 1e-4

[astar, ier] = bisection(f, a, b, tol)
print('the approximate root is', astar)
print('the error message reads:', ier)
print('f(astar) =', f(astar))


f = lambda x: x ** 3 + x - 4
a = 1
b = 4
tol = 1e-3

[astar, ier] = bisection(f, a, b, tol)
print('the approximate root is', astar)
print('the error message reads:', ier)
print('f(astar) =', f(astar))

x = np.linspace(-2,8,1000)
y = [xi - 4 * np.sin(2 * xi) - 3 for xi in x]
plt.plot(x, y)
plt.axhline(y = 0)
plt.title(f'$f(x) = x - 4\sin(x) -3$')
plt.show()

def fixedpt(f, x0, tol, Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count < Nmax):
        count = count + 1
        x1 = f(x0)
        if (abs(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]



f1 = lambda x: -np.sin(2 *x) + (5 * x) / 4 - 3 / 4


Nmax = 1000
tol = 1e-10

# test f1 '''
x0 = 7
[xstar, ier] = fixedpt(f1, x0, tol, Nmax)
print('the approximate fixed point is:', xstar)
print('f1(xstar):', f1(xstar))
print('Error message reads:', ier)