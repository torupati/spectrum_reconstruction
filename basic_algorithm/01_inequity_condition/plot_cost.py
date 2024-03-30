import numpy as np
import matplotlib.pyplot as plt


A = np.array([[1, 3], [3, 1]])
def func(x1, x2):
    x = np.array([x1, x2])
    return np.dot(x,np.dot(A, x))
    #return x1**2 + 6 * x1 * x2 + x2**2

x = np.arange(-10, 10, 0.01)
y = np.arange(-10, 10, 0.01)

X, Y = np.meshgrid(x, y)

assert X.shape == Y.shape

Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = func(X[i,j],Y[i,j])

#Z = func(X, Y)

cont = plt.contour(X, Y, Z)
plt.savefig("out.png")
#plt.show()
