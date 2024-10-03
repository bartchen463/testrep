import numpy as np

def f(x, y):
    return 3 * x ** 2 - y ** 2

def g(x, y):
    return 3 * x * y ** 2 - x ** 3 - 1

x0 = np.array([1, 1])
xn = x0
iter = 0
for i in range(1000):
    iter += 1
    xn1 = xn - np.array([[1/6, 1/18], [0, 1/6]]) @ np.array([f(xn[0], xn[1]), g(xn[0], xn[1])])
    if np.linalg.norm(xn1 - xn) < 1e-10:
        break
    xn = xn1

xn
xn[1]


x0 = np.array([1, 1])
xn = x0
def J(x, y):
    return np.array([[6 * x, -2 * y], [3 * y ** 2 - 3 * x ** 2, 6 * x * y]])

x0 = np.array([1, 1])
xn = x0
iter = 0
for i in range(1000):
    iter += 1
    j = J(xn[0],xn[1])
    P = np.linalg.solve(j, -np.array([f(xn[0], xn[1]), g(xn[0], xn[1])]))
    xn1 = xn + P

    if np.linalg.norm(xn1 - xn) < 1e-10:
        break
    xn = xn1


def f(x, y, z):
    return x ** 2 + 4 * y ** 2 + 4 * z - 16
def fx(x):
    return 2 * x
def fy(y):
    return 8 * y
def fz(z):
    return 4
def d(x, y, z):
    return f(x, y, z) / (fx(x) ** 2 + fy(y) ** 2 + fz(z) ** 2)
xn = np.array([1, 1, 1])
iter = 0

for i in range(1000):
    iter += 1
    D = d(xn[0], xn[1], xn[2])
    xn1 = xn - np.array([D * fx(xn[0]), D * fy(xn[1]), D * fz(xn[2])])
    print(f(xn[0], xn[1], xn[2]))
    if np.linalg.norm(xn - xn1) < 1e-10 and abs(f(xn1[0], xn1[1], xn1[2])) < 1e-10:
        break

    xn = xn1

print("Converged to:", xn)



