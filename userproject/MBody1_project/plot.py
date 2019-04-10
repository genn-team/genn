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

max_pn = np.amax(pn["neuron"])
max_kc = np.amax(kc["neuron"])
max_lhi = np.amax(lhi["neuron"])
max_dn = np.amax(dn["neuron"])

fig, axis = plt.subplots()


axis.vlines(np.arange(0.0, 5000.0, 100.0), 0, max_pn + max_kc + max_lhi + max_dn,
            color="gray", linestyle="--", zorder=-1, linewidth=0.2)

actors = [axis.scatter(pn["time"], pn["neuron"], s=1),
          axis.scatter(kc["time"], kc["neuron"] + max_pn, s=1),
          axis.scatter(lhi["time"], lhi["neuron"] + max_pn + max_kc, s=1),
          axis.scatter(dn["time"], dn["neuron"] + max_pn + max_kc + max_lhi, s=1)]

axis.set_xlabel("Time [ms]")
axis.set_ylabel("Neuron number")

fig.legend(actors, ["Projection neurons", "Kenyon cells", "Lateral horn interneurons", "Decision neurons"],
           ncol=2, loc="lower center")
fig.tight_layout(pad=0, rect=[0.0, 0.15, 1.0, 1.0])
plt.show()
