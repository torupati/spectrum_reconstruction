from scipy import signal
import numpy as np

original = [0, 1, 1, 2, 1, 1, 0, 0]
print(f"org[{len(original)}]={original}")
impulse_response = [0.5, -1.0, 0.5]
input_sig = original + 0.1 * np.random.randn(len(original))
recorded = signal.convolve(impulse_response, input_sig) 
#recorded = recorded + 0.01 * np.random.randn(len(recorded))
print(f"rec[{len(recorded)}]={recorded}")
recovered, remainder = signal.deconvolve(recorded, impulse_response)
print(recovered)
print(remainder)

import matplotlib.pyplot as plt
plt.close("all")
fig, ax = plt.subplots(1, 1)
ax.plot(original, label="orig")
ax.plot(input_sig, label="orig+noise")
ax.plot(recovered, label="recov")

plt.show()

