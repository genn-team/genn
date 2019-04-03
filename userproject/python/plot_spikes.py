import numpy as np
import matplotlib.pyplot as plt
import sys

def plot(filename, time_range, neuron_range, axis, yoffset=0):
    # Load data,  transposing each column into a seperate array
    data = np.loadtxt(filename, dtype=[("time", float), ("neuron", int)], unpack=True)

    # If a time or neuron range were specified
    if time_range is not None or neuron_range is not None:
        # Create a mask (initially all valid)
        mask = np.ones(data[0].shape, dtype=bool)

        # If a time range is specified, and it with mask
        if time_range is not None:
            mask &= ((data[0] >= time_range[0]) & (data[0] < time_range[1]))

        # If a neuron range is specified, and it with mask
        if neuron_range is not None:
            mask &= ((data[1] >= neuron_range[0]) & (data[1] < neuron_range[1]))

        # Apply mask
        data[0] = data[0][mask]
        data[1] = data[1][mask]

    # Plot spikes
    return axis.scatter(data[0], data[1] + yoffset, s=1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: plot_spikes.py filename [min_time max_time] [min_neuron max_neuron]")
    else:
        # Parse time range
        time_range = None
        if len(sys.argv) > 3:
            time_range = (float(sys.argv[2]), float(sys.argv[3]))

        # Parse neuron range
        neuron_range = None
        if len(sys.argv) > 5:
            neuron_range = (int(sys.argv[4]), int(sys.argv[5]))

        # Plot and show figure
        fig, axis = plt.subplots()
        plot(sys.argv[1], time_range, neuron_range, axis)
        axis.set_xlabel("Time [ms]")
        axis.set_ylabel("Neuron number")
        plt.show()
