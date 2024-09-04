import numpy as np
import matplotlib.pyplot as plt



x = np.linspace(0, 10, 100)
y = np.arange(0, 10, 0.1)

print(f'The first 3 values of x are {x[:3]}')

w = 10**(-np.linspace(1,10,10))

x = np.arange(1, len(w)+1)

plt.semilogy(x,w)
plt.title("Semi log plot of x vs w")
plt.xlabel("x")
plt.ylabel("Log w")
plt.show()

