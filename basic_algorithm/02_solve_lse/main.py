import numpy as np
import matplotlib.pyplot as plt

M, D = 2, 4
A = np.array([[100, 100, 200, 300], [100, 100, 1, 1]])
x  = np.array([-1.0, -3.0, 1.0, 2.0])
lam = np.zeros(D)
sig = 0.0 #noise density

assert A.shape == (M,D)
assert x.shape[0] == D

print(f"A={A.shape}, x={x.shape} sig={sig}")
y = np.dot(A, x) + sig * np.random.randn(M)
assert y.shape[0] == M

print(f"x0={x}")
print(f'y={y}')

def gradient(A, lam, y, x):
    print(np.dot(A.T, np.dot(A, x) - y).shape)
    print(lam.shape)
    return np.dot(A.T, np.dot(A, x) - y) - lam

def cost_func(A, y, x):
    assert A.shape == (M, D)
    assert y.shape[0] == M
    assert x.shape[0] == D
    err = y - np.dot(A, x)
    return np.dot(err, err)

def check_KKT(lam, x):
    for i in range(D):
        print(f"i={i} {lam[i]} {x[i]} {lam[i]*x[i]}")

x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, y) + lam)

print(f"x1={x} grad f(x)={gradient(A, lam, y, x)}")
check_KKT(lam, x)

x2 = np.zeros(D)
for i in range(D):
    if x[i] >= 0.0:
        x2[i] = x[i]
    else:
        x2[i] = 0.0

print(f"x2={x2} grad f(x)={gradient(A, lam, y, x2)}")
check_KKT(lam, x2)

fig, ax = plt.subplots(1, 1)
xidx = np.array([[v, 0, x2[2], x2[3]] for v in np.arange(0, 10, 0.1)])
z = [ cost_func(A, y, _x) for _x in xidx]
print(z)
ax.plot(xidx, z)
ax.set_xlabel(r"x_0")

# ------
x_grid, y_grid = np.meshgrid(np.arange(-10, 10, 0.01), np.arange(-10, 10, 0.1))
assert x_grid.shape == y_grid.shape
Z = np.zeros(x_grid.shape)
for i in range(x_grid.shape[0]):
    for j in range(x_grid.shape[1]):
        Z[i,j] = cost_func(A, y, np.array([x_grid[i,j],y_grid[i,j], x2[2], x2[3]]))

Z0 = np.zeros(x_grid.shape)
for i in range(x_grid.shape[0]):
    for j in range(x_grid.shape[1]):
        Z0[i,j] = cost_func(A, y, np.array([x_grid[i,j],y_grid[i,j], 0.0, 0.0]))


# ------
fig, axes = plt.subplots(2, 1)
ax = axes[0]
#cont = ax.contour(x_grid, y_grid, Z)
cont = ax.contourf(x_grid, y_grid, Z, levels=15)
#cont = ax.contourf(x_grid, y_grid, Z, cmap="gray", levels=13)
ax.clabel(cont, inline=True)
fig.colorbar(cont, ax=ax)

ax = axes[1]
cont = ax.contourf(x_grid, y_grid, Z0, levels=15)
#cont = ax.contourf(x_grid, y_grid, Z, cmap="gray", levels=13)
ax.clabel(cont, inline=True)
fig.colorbar(cont, ax=ax)

plt.savefig("contour.png")
ax.grid(True)
plt.show(block=False)

