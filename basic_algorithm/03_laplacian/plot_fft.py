import matplotlib.pyplot as plt
import numpy as np

N = 32 # 1024
h = np.zeros(N)
#h[0] = 1.0
h[0:2] = [1.0, -1.0]
#h[0:3] = [-0.5, 1, -0.5]
#h[0:3] = [-1.0, 1.0, -1.0]
#h[3:6] = [-0.5, 1, -0.5]
#h[0:4] = [1.0, 1.0, -1.0, -1.0]
h_steps = np.arange(N)
assert len(h) == h_steps.shape[-1]
h_sp = np.fft.fft(h)
h_freq = np.fft.fftfreq(h_steps.shape[-1])
#freq = np.fft.fftfreq(N, d=1)
#print(h_freq)

fig = plt.figure(figsize=(12, 6), layout='constrained')
fig.suptitle(f"Filter Gain (N={N})")
axs = fig.subplot_mosaic([["filter", "magnitude", "inv_mag"]])

axs["filter"].set_title(r"Filter $a[i]$")
axs["filter"].stem(h_steps[0:N+1], h[0:N+1])
axs["filter"].set_xlabel(r"$i$")
axs["filter"].set_ylim([-1.1, 1.1])
axs["filter"].grid(True)

axs["magnitude"].set_title(r"$|A[k]|^2$")
#pwr = np.abs(h_freq) ** 2
pwr = (h_freq * h_freq.conj()).real
#axs["magnitude"].plot(h_freq[0:N//2+1], np.absolute(h_sp)[0:N//2+1], markersize=1)
axs["magnitude"].plot(np.absolute(h_sp)[0:N//2+1], "o", markersize=5)
axs["magnitude"].grid(True)
#axs["magnitude"].plot([v/N for v in range(N//2)], pwr[0:N//2]/N)
#axs["magnitude"].plot(h_freq, np.absolute(h_sp), ".", markersize=1, label=r"$|A(\Omega)|$")
#axs["magnitude"].plot(h_freq, h_sp.real, ".", markersize=1, label=r"$\Re(A(\Omega))$")
#axs["magnitude"].plot(h_freq, h_sp.imag, ".", markersize=1, label=r"$\Im(A(\Omega))$")
#axs["magnitude"].legend()
axs["magnitude"].set_xlabel(r"$k$")
axs["magnitude"].set_xticks([0, (N//2)//2, N//2])
axs["magnitude"].set_xticklabels([f"{i}" for i in [0, (N//2)//2, N//2]])
#axs["magnitude"].set_xlim([0,0.5])
#axs["phase"].magnitude_spectrum(h)

#axs["log_magnitude"].magnitude_spectrum(h, scale='dB')
#axs["log_magnitude"].plot([i / len(h_freq) for i in range(int(len(h_freq)/2)], 10.0 * np.log10(np.abs(h_freq)/len(h_freq)))

axs["inv_mag"].set_title(r"$\frac{1}{|A[k]|^2}$")
#axs["inv_mag"].plot(h_freq[1:N//2], [1.0/pwr[i] for i in range(1,N//2)])
axs["inv_mag"].plot([i for i in range(1,N//2+1)], 1.0/np.absolute(h_sp)[1:N//2+1], "o", markersize=5)
#axs["inv_mag"].set_ylim([0,10])
axs["inv_mag"].grid(True)
#axs["inv_mag"].set_xlabel(r"$\Omega$")
axs["inv_mag"].set_xlabel(r"$k$")
axs["inv_mag"].set_xticks([0, (N//2)//2, N//2])
axs["inv_mag"].set_xticklabels([f"{i}" for i in [0, (N//2)//2, N//2]])
#axs["inv_mag"].set_xlim([0,0.5])
#axs["phase"].plot([1.0 / v for v in np.abs(h_freq)])

plt.savefig("filter_power.png")
