import numpy as np
import matplotlib.pyplot as plt
import sys

def plot(filename, time_range, neuron_range):
    # If a neuron range is specified, convert this into a list of columns to parse
    cols = None if neuron_range is None else [0,] + range(1 + neuron_range[0], 1 + neuron_range[1])

    # Load data,  transposing each column into a seperate array
    data = np.loadtxt(filename, dtype=float, unpack=True, usecols=cols)

    # If a time range were specified
    if time_range is not None :
        mask = ((data[0] >= time_range[0]) & (data[0] < time_range[1]))

        # Apply mask
        data = data[:,mask]

    # Plot spikes
    fig, axis = plt.subplots()
    for v in data[1:]:
        axis.plot(data[0], v)

    axis.set_xlabel("Time [ms]")
    axis.set_ylabel("Voltage")
    return fig, axis


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: plot_voltages.py filename [min_time max_time] [min_neuron max_neuron]")
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
        plot(sys.argv[1], time_range, neuron_range)
        plt.show()
