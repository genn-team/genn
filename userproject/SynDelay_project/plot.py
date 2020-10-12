import matplotlib.pyplot as plt
import numpy as np
import sys

assert len(sys.argv) >= 2

def load_spikes(filename):
    return np.loadtxt(filename, dtype={"names": ("time", "neuron_id"),
                                       "formats": (np.float, np.int)})

def load_voltages(filename):
    return np.loadtxt(filename, dtype={"names": ("time", "v_input", "v_inter", "v_output"),
                                       "formats": (np.float, np.float, np.float, np.float)})
    
# Load spikes
input_spikes = load_spikes(sys.argv[1] + "_input_st")
inter_spikes = load_spikes(sys.argv[1] + "_inter_st")
output_spikes = load_spikes(sys.argv[1] + "_output_st")
voltages = load_voltages(sys.argv[1] + "_Vm")

# Create plot
figure, axes = plt.subplots(3, sharex=True)

input_v = axes[0].twinx()
inter_v = axes[1].twinx()
output_v = axes[2].twinx()

axes[0].scatter(input_spikes["time"], input_spikes["neuron_id"], s=2)
input_v.plot(voltages["time"], voltages["v_input"], color="red")
axes[1].scatter(inter_spikes["time"], inter_spikes["neuron_id"], s=2)
inter_v.plot(voltages["time"], voltages["v_inter"], color="red")
axes[2].scatter(output_spikes["time"], output_spikes["neuron_id"], s=2)
output_v.plot(voltages["time"], voltages["v_output"], color="red")

axes[0].set_ylabel("Input neuron number")
input_v.set_ylabel("Input neuron membrane voltage")
axes[1].set_ylabel("Inter neuron number")
inter_v.set_ylabel("Inter neuron membrane voltage")
axes[2].set_ylabel("Output neuron number")
output_v.set_ylabel("Output neuron membrane voltage")
axes[2].set_xlabel("Time [ms]")

plt.show()