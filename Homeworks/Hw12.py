import numpy as np

n = 4
A = np.zeros(shape=(n,n))
for i in range(n):
    for j in range(n):
        A[i,j] = 1 / ((i + 1) + (j + 1) - 1)

b = np.random.uniform(size = n)
b = b / np.linalg.norm(b, np.inf)

