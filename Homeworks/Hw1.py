import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1.92, 2.08, 0.001)

def f_Exp(x):
    y = [0]*len(x)
    for i in range(len(x)):
        xi = x[i]
        y[i] = xi**9 - 18*xi**8 + 144*xi**7 - 672*xi**6 + 2016*xi**5\
                -4032*xi**4 + 5376*xi**3 - 4608*xi**2 + 2304*xi - 512
    return y

plt.plot(x, f_Exp(x))
plt.title('p(x) evaluated using full coefficient expansion')
plt.show()

def f_Fact(x):
    y = [0]*len(x)
    for i in range(len(x)):
        xi = x[i]
        y[i] = (xi - 2)**9
    return y

plt.plot(x, f_Fact(x))
plt.title('p(x) evaluated using factored form')
plt.show()


delta = [10**(i) for i in range(-16,1)]

x = np.pi
diff = [0]*len(delta)
for i in range(len(delta)):
    d = delta[i]
    diff[i] = np.cos(x+d) - np.cos(x) + 2*np.sin((2*x+d)/2)*np.sin(d/2)

plt.plot(delta, diff)
plt.title("$\cos(\pi+\delta) - \cos(\pi) + 2\sin((2\pi+\delta)/2)\sin(\delta/2)$")
plt.xscale('log')
plt.xlabel("$\delta$")
plt.ylabel("Difference")
plt.show()

x = 10**6
diff = [0]*len(delta)
for i in range(len(delta)):
    d = delta[i]
    diff[i] = np.cos(x+d) - np.cos(x) + 2*np.sin((2*x+d)/2)*np.sin(d/2)

plt.plot(delta, diff)
plt.title("$\cos(10^6+\delta) - \cos(10^6) + 2\sin((2*10^6+\delta)/2)\sin(\delta/2)$")
plt.xscale('log')
plt.xlabel("$\delta$")
plt.ylabel("Difference")
plt.show()

x = 10**6
diff = [0]*len(delta)
for i in range(len(delta)):
    d = delta[i]
    diff[i] = np.cos(x+d) - np.cos(x) + d*np.sin(x) + d**2/2*np.sin(x+d)