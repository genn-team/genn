import matplotlib.pyplot as plt
import numpy as np

state = np.loadtxt("spikes.csv", delimiter=",")

# Create plot
figure, axes = plt.subplots(2, sharex=True)

# Compute histograms, dividing 1s simulation into 100 10ms bins
rate_hist, rate_bins = np.histogram(state[:,0], bins=100)

# Convert histogram into Hz (spikes per 1000ms)
rate_hist = np.multiply(rate_hist, 1000.0 / 10.0, dtype=float)

# Convert histogram into average per neuron
rate_hist /= 10000.0

axes[0].scatter(state[:,0], state[:,1], s=1)
axes[1].bar(rate_bins[:-1], rate_hist, 10.0)

axes[0].set_ylabel("Neuron number")
axes[1].set_xlabel("Time [ms]")
axes[1].set_ylabel("Average rate [Hz]")

# Show plot
plt.show()
