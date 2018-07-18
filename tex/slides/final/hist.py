import numpy as np
import matplotlib.pyplot as plt

n = 1000
x = np.linspace(0, 255, 255)
y = np.random.randint(0, 255, int(n))
y = np.append(y, np.random.normal(255, 5, int(n * 0.2)))
y = np.append(y, np.random.normal(0, 2, int(n * 0.04)))

fig, ax = plt.subplots()

ax.hist(y, bins=60, density=True, histtype='step')
ax.set_xlabel("RGB-Wert")
ax.set_ylabel("Dichte")
ax.set_title("Histogramm des Blaukanals")

fig.tight_layout(pad=0)
fig.savefig('content/hist.pdf')
