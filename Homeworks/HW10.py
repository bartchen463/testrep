import numpy as np
import matplotlib.pyplot as plt

def T6(x):
    return x - x**3/6 + x**5/120

def P33(x):
    return (x - (7/60)*x**3)/(1 + x**2/20)
def P24(x):
    return x / (1 + x**2/6 + 7*x**4/360)
def P42(x):
    return (x - (7/60)*x**3) / (1 + x**2/20)

xvals = np.linspace(0, 5, 1000)
exact = np.sin(xvals)
T6approx = T6(xvals)
P33approx = P33(xvals)
P24approx = P24(xvals)
P42approx = P42(xvals)

plt.plot(xvals,exact - T6approx, label = "Degree 6 Maclaurin Error")
plt.plot(xvals, exact - P33approx, label = "Pade 3, 3 Error")
plt.plot(xvals, exact - P24approx, label = "Pade 2, 4 Error")
plt.plot(xvals, exact - P42approx, label = "Pade 4, 2 Error")
plt.title("Error in approximation of sinx")
plt.legend()
plt.show()