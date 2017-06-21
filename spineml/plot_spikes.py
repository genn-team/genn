import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

with open(sys.argv[1], "rb") as spikes_csv_file:
    spikes_csv_reader = csv.reader(spikes_csv_file, delimiter = ",")
   
    # Read data and zip into columns
    spikes_data_columns = zip(*spikes_csv_reader)

    # Convert CSV columns to numpy
    spike_times = np.asarray(spikes_data_columns[0], dtype=float)
    spike_neuron_id = np.asarray(spikes_data_columns[1], dtype=int)
    
    # Determine ranges
    max_time = np.amax(spike_times);
    max_neuron = np.amax(spike_neuron_id)
    
    # Create plot
    figure, axes = plt.subplots(2, sharex=True)

    # Plot spikes
    axes[0].scatter(spike_times, spike_neuron_id, s=2, edgecolors="none")

    # Plot rates
    # **NOTE** using max_neuron here potentially isn't quite right
    bins = np.arange(0, max_time + 1, 10)
    rate = np.histogram(spike_times, bins=bins)[0] * (1000.0 / 10.0) * (1.0 / float(max_neuron))
    axes[1].plot(bins[0:-1], rate)

    axes[0].set_title("Spikes")
    axes[1].set_title("Firing rates")

    axes[0].set_ylabel("Neuron number")
    axes[1].set_ylabel("Mean firing rate [Hz]")
    
    axes[1].set_xlabel("Time [ms]")
    
    axes[0].set_ylim((0, max_neuron))
    axes[1].set_xlim((0, max_time))
    
    # Show plot
    plt.show()

