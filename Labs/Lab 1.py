import numpy as np
import matplotlib.pyplot as plt


## This makes a vector of 100 evenly spaced points from 0 to 10
x = np.linspace(0, 10, 100)
## This is a vector of 100 numbers starting from 0 and increasing by 0.1
y = np.arange(0, 10, 0.1)

## Print the first 3 vales of x
print(f'The first 3 values of x are {x[:3]}')


## Ten values of 10**(-x) for integers x between 1 and 10
w = 10**(-np.linspace(1,10,10))


## The integers from 1 to 10
x =(np.arange(1, len(w)+1))


## Plotting x against w on a logarithmic scale
plt.semilogy(x,w)
plt.title("Semi log plot of x vs w")
plt.xlabel("x")
plt.ylabel("Log w")
plt.show()

