import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

with open(sys.argv[1], "rb") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",")

    # Skip headers
    csv_reader.next()

    # Read data and zip into columns
    data_columns = zip(*csv_reader)

    # Convert times to numpy
    times = np.asarray(data_columns[0], dtype=float)
    neuron_id = np.asarray(data_columns[1], dtype=int)
    value = np.asarray(data_columns[2], dtype=float)
    
    unique_neuron_ids = np.unique(neuron_id)
    neuron_masks = [neuron_id == n for n in unique_neuron_ids]
    
    # Create plot
    figure, axis = plt.subplots()
    axis.set_xlabel("time [ms]")
    axis.set_ylabel("value")
    
    # Plot voltages
    for m in neuron_masks:
        axis.plot(times[m], value[m])

    # Show plot
    plt.show()
