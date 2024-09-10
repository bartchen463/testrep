import numpy as np
import math
import matplotlib.pyplot as plt
import random
A = np.array([[1,1], [1+1e-10, 1-1e-10]])*(1/2)
U, S, Vt = np.linalg.svd(A)
S[0]/S[1]

## Condition Number of A
np.linalg.cond(A)/1e10
A_inv = np.array([[1-1e10, 1e10], [1+1e10, -1e10]])

## Check relative error
db = np.array([2e-5, -6e-5])
((np.linalg.norm(A_inv@db))/np.linalg.norm(A_inv@np.array([1,1])))

## 3C
x = 9.999999995000000*1e-10
def f(x):
    y = math.e**x
    return y - 1

f(x)

(1e-9 - f(x))/1e-9

def err(n):
    return (1/(math.factorial(n+1)))*(x**(n+1))*math.e**x

## 2 Terms is good enough
err(2)

y = x + (x**2)/2

## Zero relative error (machine precision)
(1e-9 - y)/1e-9

## Q4a
t = np.arange(0, np.pi+np.pi/60, np.pi/30)
y = np.cos(t)
S = 0
for i in range(len(t)):
    S += t[i]*y[i]
print(f'the sum is: {S}')

## b
def plot_b(R, dr, f, p):
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = R*(1+dr*(np.sin(f*theta + p)))*np.cos(theta)
    y = R*(1+dr*(np.sin(f*theta + p)))*np.sin(theta)
    plt.axis('equal')
    plt.plot(x,y)
R = 1.2
dr = 0.1
f = 15
p = 0
plot_b(R, dr, f, p)
plt.title("Single Parametric Curve")
plt.show()

for i in range(1,11):
    R = i
    dr = 0.05
    f = 2 + i
    p = random.uniform(0,2)
    plot_b(R, dr, f, p)

plt.title("10 Parametric Curves")
plt.show()