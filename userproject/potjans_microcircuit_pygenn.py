import numpy as np 
import matplotlib.pyplot as plt 

from pygenn import (GeNNModel, VarLocation, SpanType, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
from scipy.stats import norm
from six import iteritems, itervalues
from time import perf_counter

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# Layer names
LAYER_NAMES = ["23", "4", "5", "6"]

# Population names
POPULATION_NAMES = ["E", "I"]

# Simulation timestep [ms]
DT_MS = 0.1

# Simulation duration [ms]
DURATION_MS = 1000.0

# Should kernel timing be measured?
MEASURE_TIMING = True

# Should we use procedural rather than in-memory connectivity?
PROCEDURAL_CONNECTIVITY = False

# Should we rebuild the model rather than loading previous version
BUILD_MODEL = True

# How many threads to use per spike for procedural connectivity?
NUM_THREADS_PER_SPIKE = 8

# Scaling factors for number of neurons and synapses
NEURON_SCALING_FACTOR = 1.0
CONNECTIVITY_SCALING_FACTOR = 1.0

# Background rate per synapse
BACKGROUND_RATE = 8.0  # spikes/s

# Relative inhibitory synaptic weight
G = -4.0

# Mean synaptic weight for all excitatory projections except L4e->L2/3e
MEAN_W = 87.8e-3  # nA
EXTERNAL_W = 87.8e-3   # nA

# Mean synaptic weight for L4e->L2/3e connections
# See p. 801 of the paper, second paragraph under 'Model Parameterization',
# and the caption to Supplementary Fig. 7
LAYER_23_4_W = 2.0 * MEAN_W   # nA

# Standard deviation of weight distribution relative to mean for
# all projections except L4e->L2/3e
REL_W = 0.1

# Standard deviation of weight distribution relative to mean for L4e->L2/3e
# This value is not mentioned in the paper, but is chosen to match the
# original code by Tobias Potjans
LAYER_23_4_RELW = 0.05

# Numbers of neurons in full-scale model
NUM_NEURONS = {
    "23":   {"E":20683, "I": 5834},
    "4":    {"E":21915, "I": 5479},
    "5":    {"E":4850,  "I": 1065},
    "6":    {"E":14395, "I": 2948}}

# Probabilities for >=1 connection between neurons in the given populations.
# The first index is for the target population; the second for the source population
CONNECTION_PROBABILTIES = {
    "23E":  {"23E": 0.1009, "23I": 0.1689,  "4E": 0.0437,   "4I": 0.0818,   "5E": 0.0323,   "5I": 0.0,      "6E": 0.0076,   "6I": 0.0},
    "23I":  {"23E": 0.1346, "23I": 0.1371,  "4E": 0.0316,   "4I": 0.0515,   "5E": 0.0755,   "5I": 0.0,      "6E": 0.0042,   "6I": 0.0},
    "4E":   {"23E": 0.0077, "23I": 0.0059,  "4E": 0.0497,   "4I": 0.135,    "5E": 0.0067,   "5I": 0.0003,   "6E": 0.0453,   "6I": 0.0},
    "4I":   {"23E": 0.0691, "23I": 0.0029,  "4E": 0.0794,   "4I": 0.1597,   "5E": 0.0033,   "5I": 0.0,      "6E": 0.1057,   "6I": 0.0},
    "5E":   {"23E": 0.1004, "23I": 0.0622,  "4E": 0.0505,   "4I": 0.0057,   "5E": 0.0831,   "5I": 0.3726,   "6E": 0.0204,   "6I": 0.0},
    "5I":   {"23E": 0.0548, "23I": 0.0269,  "4E": 0.0257,   "4I": 0.0022,   "5E": 0.06,     "5I": 0.3158,   "6E": 0.0086,   "6I": 0.0},
    "6E":   {"23E": 0.0156, "23I": 0.0066,  "4E": 0.0211,   "4I": 0.0166,   "5E": 0.0572,   "5I": 0.0197,   "6E": 0.0396,   "6I": 0.2252},
    "6I":   {"23E": 0.0364, "23I": 0.001,   "4E": 0.0034,   "4I": 0.0005,   "5E": 0.0277,   "5I": 0.008,    "6E": 0.0658,   "6I": 0.1443}}
    

# In-degrees for external inputs
NUM_EXTERNAL_INPUTS = {
    "23":   {"E": 1600, "I": 1500},
    "4":    {"E": 2100, "I": 1900},
    "5":    {"E": 2000, "I": 1900},
    "6":    {"E": 2900, "I": 2100}}

# Mean rates in the full-scale model, necessary for scaling
# Precise values differ somewhat between network realizations
MEAN_FIRING_RATES = {
    "23":   {"E": 0.971,    "I": 2.868},
    "4":    {"E": 4.746,    "I": 5.396},
    "5":    {"E": 8.142,    "I": 9.078},
    "6":    {"E": 0.991,    "I": 7.523}}

# Means and standard deviations of delays from given source populations (ms)
MEAN_DELAY = {"E": 1.5, "I": 0.75}

DELAY_SD = {"E": 0.75, "I": 0.375}

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def get_scaled_num_neurons(layer, pop):
    return int(round(NEURON_SCALING_FACTOR * NUM_NEURONS[layer][pop]))

def get_full_num_inputs(src_layer, src_pop, trg_layer, trg_pop):
    num_src = NUM_NEURONS[src_layer][src_pop]
    num_trg = NUM_NEURONS[trg_layer][trg_pop]
    connection_prob = CONNECTION_PROBABILTIES[trg_layer + trg_pop][src_layer + src_pop]

    return int(round(np.log(1.0 - connection_prob) / np.log(float(num_trg * num_src - 1) / float(num_trg * num_src))) / num_trg)

def get_mean_weight(src_layer, src_pop, trg_layer, trg_pop):
    # Determine mean weight
    if src_pop == "E":
        if src_layer == "4" and trg_layer == "23" and trg_pop == "E":
            return LAYER_23_4_W
        else:
            return MEAN_W
    else:
        return G * MEAN_W

def get_scaled_num_connections(src_layer, src_pop, trg_layer, trg_pop):
    # Scale full number of inputs by scaling factor
    num_inputs = get_full_num_inputs(src_layer, src_pop, trg_layer, trg_pop) * CONNECTIVITY_SCALING_FACTOR
    assert num_inputs >= 0.0

    # Multiply this by number of postsynaptic neurons
    return int(round(num_inputs * float(get_scaled_num_neurons(trg_layer, trg_pop))))

def get_full_mean_input_current(layer, pop):
    # Loop through source populations
    mean_input_current = 0.0
    for src_layer in LAYER_NAMES:
        for src_pop in POPULATION_NAMES:
            mean_input_current += (get_mean_weight(src_layer, src_pop, layer, pop) *
                                   get_full_num_inputs(src_layer, src_pop, layer, pop) *
                                   MEAN_FIRING_RATES[src_layer][src_pop])

    # Add mean external input current
    mean_input_current += EXTERNAL_W * NUM_EXTERNAL_INPUTS[layer][pop] * BACKGROUND_RATE
    assert mean_input_current >= 0.0
    return mean_input_current

# ----------------------------------------------------------------------------
# Network creation
# ----------------------------------------------------------------------------
model = GeNNModel("float", "potjans_microcircuit")
model.dt = DT_MS
model.fuse_postsynaptic_models = True
model.default_narrow_sparse_ind_enabled = True
model.timing_enabled = MEASURE_TIMING
model.default_var_location = VarLocation.DEVICE
model.default_sparse_connectivity_location = VarLocation.DEVICE

lif_init = {"V": init_var("Normal", {"mean": -58.0, "sd": 5.0}), "RefracTime": 0.0}
poisson_init = {"current": 0.0}

exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5})

quantile = 0.9999
normal_quantile_cdf = norm.ppf(quantile)
max_delay = {pop: MEAN_DELAY[pop] + (DELAY_SD[pop] * normal_quantile_cdf)
             for pop in POPULATION_NAMES}
print("Max excitatory delay:%fms , max inhibitory delay:%fms" % (max_delay["E"], max_delay["I"]))

# Calculate maximum dendritic delay slots
# **NOTE** it seems inefficient using maximum for all but this allows more aggressive merging of postsynaptic models
max_dendritic_delay_slots = int(round(max(itervalues(max_delay)) / DT_MS))
print("Max dendritic delay slots:%d" % max_dendritic_delay_slots)

print("Creating neuron populations:")
total_neurons = 0
neuron_populations = {}
for layer in LAYER_NAMES:
    for pop in POPULATION_NAMES:
        pop_name = layer + pop

        # Calculate external input rate, weight and current
        ext_input_rate = NUM_EXTERNAL_INPUTS[layer][pop] * CONNECTIVITY_SCALING_FACTOR * BACKGROUND_RATE
        ext_weight = EXTERNAL_W / np.sqrt(CONNECTIVITY_SCALING_FACTOR)
        ext_input_current = 0.001 * 0.5 * (1.0 - np.sqrt(CONNECTIVITY_SCALING_FACTOR)) * get_full_mean_input_current(layer, pop)
        assert ext_input_current >= 0.0

        lif_params = {"C": 0.25, "TauM": 10.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh" : -50.0,
                      "Ioffset": ext_input_current, "TauRefrac": 2.0}
        poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": ext_input_rate}

        pop_size = get_scaled_num_neurons(layer, pop)
        neuron_pop = model.add_neuron_population(pop_name, pop_size, "LIF", lif_params, lif_init)
        model.add_current_source(pop_name + "_poisson", "PoissonExp", pop_name, poisson_params, poisson_init)

        # Enable spike recording
        neuron_pop.spike_recording_enabled = True

        print("\tPopulation %s: num neurons:%u, external input rate:%f, external weight:%f, external DC offset:%f" % (pop_name, pop_size, ext_input_rate, ext_weight, ext_input_current))

        # Add number of neurons to total
        total_neurons += pop_size

        # Add neuron population to dictionary
        neuron_populations[pop_name] = neuron_pop

# Loop through target populations and layers
print("Creating synapse populations:")
total_synapses = 0
num_sub_rows = NUM_THREADS_PER_SPIKE if PROCEDURAL_CONNECTIVITY else 1
for trg_layer in LAYER_NAMES:
    for trg_pop in POPULATION_NAMES:
        trg_name = trg_layer + trg_pop

        # Loop through source populations and layers
        for src_layer in LAYER_NAMES:
            for src_pop in POPULATION_NAMES:
                src_name = src_layer + src_pop

                # Determine mean weight
                mean_weight = get_mean_weight(src_layer, src_pop, trg_layer, trg_pop) / np.sqrt(CONNECTIVITY_SCALING_FACTOR)

                # Determine weight standard deviation
                if src_pop == "E" and src_layer == "4" and trg_layer == "23" and trg_pop == "E":
                    weight_sd = mean_weight * LAYER_23_4_RELW
                else:
                    weight_sd = abs(mean_weight * REL_W)

                # Calculate number of connections
                num_connections = get_scaled_num_connections(src_layer, src_pop, trg_layer, trg_pop)

                if num_connections > 0:
                    num_src_neurons = get_scaled_num_neurons(src_layer, src_pop)
                    num_trg_neurons = get_scaled_num_neurons(trg_layer, trg_pop)

                    print("\tConnection between '%s' and '%s': numConnections=%u, meanWeight=%f, weightSD=%f, meanDelay=%f, delaySD=%f" 
                          % (src_name, trg_name, num_connections, mean_weight, weight_sd, MEAN_DELAY[src_pop], DELAY_SD[src_pop]))

                    # Build parameters for fixed number total connector
                    connect_params = {"total": num_connections}

                    # Build distribution for delay parameters
                    d_dist = {"mean": MEAN_DELAY[src_pop], "sd": DELAY_SD[src_pop], "min": 0.0, "max": max_delay[src_pop]}

                    total_synapses += num_connections

                    # Build unique synapse name
                    synapse_name = src_name + "_" + trg_name

                    matrix_type = "PROCEDURAL" if PROCEDURAL_CONNECTIVITY else "SPARSE"

                    # Excitatory
                    if src_pop == "E":
                        # Build distribution for weight parameters
                        # **HACK** np.float32 doesn't seem to automatically cast 
                        w_dist = {"mean": mean_weight, "sd": weight_sd, "min": 0.0, "max": float(np.finfo(np.float32).max)}

                        # Create weight parameters
                        static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                                 {"g": init_var("NormalClipped", w_dist),
                                                                  "d": init_var("NormalClippedDelay", d_dist)})
                        # Add synapse population
                        syn_pop = model.add_synapse_population(synapse_name, matrix_type, 0,
                            neuron_populations[src_name], neuron_populations[trg_name],
                            static_synapse_init, exp_curr_init,
                            init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))

                        # Set max dendritic delay and span type
                        syn_pop.max_dendritic_delay_timesteps = max_dendritic_delay_slots

                        if PROCEDURAL_CONNECTIVITY:
                            syn_pop.span_type = SpanType.PRESYNAPTIC
                            syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE
                    # Inhibitory
                    else:
                        # Build distribution for weight parameters
                        # **HACK** np.float32 doesn't seem to automatically cast 
                        w_dist = {"mean": mean_weight, "sd": weight_sd, "min": float(-np.finfo(np.float32).max), "max": 0.0}

                        # Create weight parameters
                        static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                                 {"g": init_var("NormalClipped", w_dist),
                                                                  "d": init_var("NormalClippedDelay", d_dist)})
                        # Add synapse population
                        syn_pop = model.add_synapse_population(synapse_name, matrix_type, 0,
                            neuron_populations[src_name], neuron_populations[trg_name],
                            static_synapse_init, exp_curr_init,
                            init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))

                        # Set max dendritic delay and span type
                        syn_pop.max_dendritic_delay_timesteps = max_dendritic_delay_slots

                        if PROCEDURAL_CONNECTIVITY:
                            syn_pop.span_type = SpanType.PRESYNAPTIC
                            syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE
print("Total neurons=%u, total synapses=%u" % (total_neurons, total_synapses))

if BUILD_MODEL:
    print("Building Model")
    model.build()

print("Loading Model")
duration_timesteps = int(round(DURATION_MS / DT_MS))
ten_percent_timestep = duration_timesteps // 10
model.load(num_recording_timesteps=duration_timesteps)

print("Simulating")

# Loop through timesteps
sim_start_time = perf_counter()
while model.t < DURATION_MS:
    # Advance simulation
    model.step_time()

    # Indicate every 10%
    if (model.timestep % ten_percent_timestep) == 0:
        print("%u%%" % (model.timestep / 100))
        
sim_end_time =  perf_counter()


# Download recording data
model.pull_recording_buffers_from_device()

print("Timing:")
print("\tSimulation:%f" % ((sim_end_time - sim_start_time) * 1000.0))

if MEASURE_TIMING:
    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tSynapse simulation:%f" % (1000.0 * model.presynaptic_update_time))


# Create plot
figure, axes = plt.subplots(1, 2)

# **YUCK** re-order neuron populationsf for plotting
ordered_neuron_populations = list(reversed(list(itervalues(neuron_populations))))

start_id = 0
bar_y = 0.0
for pop in ordered_neuron_populations:
    # Get recording data
    spike_times, spike_ids = pop.spike_recording_data[0]
    
    # Plot spikes
    actor = axes[0].scatter(spike_times, spike_ids + start_id, s=2, edgecolors="none")

    # Plot bar showing rate in matching colour
    axes[1].barh(bar_y, len(spike_times) / (float(pop.size) * DURATION_MS / 1000.0), 
                 align="center", color=actor.get_facecolor(), ecolor="black")

    # Update offset
    start_id += pop.size

    # Update bar pos
    bar_y += 1.0

axes[0].set_xlabel("Time [ms]")
axes[0].set_ylabel("Neuron number")

axes[1].set_xlabel("Mean firingrate [Hz]")
axes[1].set_yticks(np.arange(0.0, len(neuron_populations), 1.0))
axes[1].set_yticklabels([n.name for n in ordered_neuron_populations])

# Show plot
plt.show()

