import matplotlib.pyplot as plt
import numpy as np
from os import path
import sys

assert len(sys.argv) == 2

data_dir = sys.argv[1] + "_output"

# Load data
pn = np.loadtxt(path.join(data_dir, sys.argv[1] + ".pn.st"),
                dtype=[("time", float), ("neuron", int)])
kc = np.loadtxt(path.join(data_dir, sys.argv[1] + ".kc.st"),
                dtype=[("time", float), ("neuron", int)])
lhi = np.loadtxt(path.join(data_dir, sys.argv[1] + ".lhi.st"),
                dtype=[("time", float), ("neuron", int)])
dn = np.loadtxt(path.join(data_dir, sys.argv[1] + ".dn.st"),
                dtype=[("time", float), ("neuron", int)])

# Determine (approximately) how many neurons are in each population
max_pn = np.amax(pn["neuron"])
max_kc = np.amax(kc["neuron"])
max_lhi = np.amax(lhi["neuron"])
max_dn = np.amax(dn["neuron"])

# Determine (approximately) the duration of the simulation
duration = max(np.amax(pn["time"]), np.amax(kc["time"]),
               np.amax(lhi["time"]), np.amax(dn["time"]))

# Build bins corresponding to pattern presentations
pattern_bins = np.arange(0.0, duration + 99.0, 100.0)
end_of_last_pattern = pattern_bins[-1]

fig, axes = plt.subplots(3, sharex=True,
                         gridspec_kw={"height_ratios":[4, 1, 1]})

# Draw pattern bins on raster plot axis
axes[0].vlines(pattern_bins, 0, max_pn + max_kc + max_lhi + max_dn,
            color="gray", linestyle="--", zorder=-1, linewidth=0.2)

# Draw raster plots for different populations
actors = [axes[0].scatter(pn["time"], pn["neuron"], s=1),
          axes[0].scatter(kc["time"], kc["neuron"] + max_pn, s=1),
          axes[0].scatter(lhi["time"], lhi["neuron"] + max_pn + max_kc, s=1),
          axes[0].scatter(dn["time"], dn["neuron"] + max_pn + max_kc + max_lhi, s=1)]

# Draw histograms of number of PNs and KCs active for each pattern
axes[1].hist(pn["time"], bins=pattern_bins)
axes[2].hist(kc["time"], bins=pattern_bins)

# Label axes
axes[2].set_xlabel("Time [ms]")
axes[0].set_ylabel("Neuron number")
axes[1].set_ylabel("Active\nPNs")
axes[2].set_ylabel("Active\nKCs")

axes[0].set_xlim((0.0, end_of_last_pattern))

# Add raster plot colour legend
fig.legend(actors, ["Projection neurons", "Kenyon cells", "Lateral horn interneurons", "Decision neurons"],
           ncol=2, loc="lower center")
fig.tight_layout(pad=0, rect=[0.0, 0.15, 1.0, 1.0])
plt.show()
