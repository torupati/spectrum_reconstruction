import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

plt.close('all')

x_min, x_max = 0, 500


def gaussian(x, mu, sig2):
    return (
        1.0 / (np.sqrt(2.0 * np.pi * sig2)) * np.exp(- (1/2.0) * (x - mu) ** 2 / sig2)
    )

def log_normal(x, mu, sig2) -> float:
    """_summary_

    Args:
        x (_type_): _description_
        mu (_type_): _description_
        sig2 (_type_): _description_

    Returns:
        float: _description_
    """
    return (
        1.0 / (np.sqrt(2.0 * np.pi * sig2) * x) 
        * np.exp(- (1/2.0) * (np.log(x) - mu) ** 2 / sig2)
    )


centers = [100, 200, 400]
M = 3

x_index = np.arange(x_min, x_max, 1)
h_bank = np.ones((M, len(x_index))) * np.nan
for m in range(M):
    h_bank[m, :] = [gaussian(x, centers[m], 1600) for x in x_index]
y = [10* gaussian(x, 0, 300**2) for x in x_index]

fig, ax = plt.subplots(1, 1)
for m in range(M):
    #ax.plot(x_index, [log_normal(x, np.log(_x0), 1000) for x in x_index],
    #        label=r'$\mu={0}$'.format(_x0))
    ax.plot(x_index, h_bank[m, :], label=r'$h_{0}(x)$'.format(m))
ax.plot(x_index, y, label="y")
ax.legend(loc='upper right')
fig.savefig('out.png')

# ------ (filterbank output)
b_obs = [np.dot(y, h_bank[m, :]) for m in range(M)]
print("b_obs=", b_obs)

# ------ calculate matrix H
H = np.ones((M, M)) * np.nan
for m1 in range(M):
    for m2 in range(M):
        H[m1, m2] = np.dot(h_bank[m1,:], h_bank[m2, :])
print(H)

# ------ (estimated coeefieicnet)
a_est = np.dot(np.linalg.inv(H), b_obs)
print("a_est=", a_est)

# ------ reconstruct
y_est = np.zeros(len(x_index))
for m in range(M):
    y_est = y_est + a_est[m] * h_bank[m, :]
print(y_est.shape)

fig, ax = plt.subplots(1, 1)
for m in range(M):
    ax.plot(x_index, a_est[m] * h_bank[m, :], label=r'$\alpha_{0} h_{1}(x)$'.format(m, m))
ax.plot(x_index, y, label="y (obs)")
ax.plot(x_index, y_est, label='y (recon)')
ax.legend(loc='upper right')
fig.savefig('out2.png')

print('---eigenvalues')
w, v = LA.eig(H)
print(w, v)
idx = np.argsort(w)
w, v = w[idx], v[:, idx]
print(w, v)

