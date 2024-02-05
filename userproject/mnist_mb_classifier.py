"""
MNIST classification using an insect-inspired mushroom body model
=================================================================

This example doesn't do much, it just makes a simple plot
"""
import mnist
import numpy as np
from copy import copy
from argparse import ArgumentParser
from pygenn import (create_current_source_model, create_neuron_model,
                    create_weight_update_model, init_sparse_connectivity,
                    init_postsynaptic, init_weight_update, GeNNModel)
from tqdm.auto import tqdm

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# Simulation time step
DT = 0.1

# Scaling factor for converting normalised image pixels to input currents (nA)
INPUT_SCALE = 80.0

# Size of current to use to stimulate correct MBON when training (nA)
MBON_STIMULUS_CURRENT = 5.0

# Number of Projection Neurons in model (should match image size)
NUM_PN = 784

# Number of Kenyon Cells in model (defines memory capacity)
NUM_KC = 20000

# Number of output neurons in model
NUM_MBON = 10

# How long to present each image to model
PRESENT_TIME_MS = 20.0

# Standard LIF neurons parameters
LIF_PARAMS = {
    "C": 0.2,
    "TauM": 20.0,
    "Vrest": -60.0,
    "Vreset": -60.0,
    "Vthresh": -50.0,
    "Ioffset": 0.0,
    "TauRefrac": 2.0}

# We only want PNs to spike once
PN_PARAMS = copy(LIF_PARAMS)
PN_PARAMS["TauRefrac"] = 100.0

# Weight of each synaptic connection
PN_KC_WEIGHT = 0.2

# Time constant of synaptic integration
PN_KC_TAU_SYN = 3.0

# How many projection neurons should be connected to each Kenyon Cell
PN_KC_FAN_IN = 20

# We will use weights of 1.0 for KC->GGN connections and
# want the GGN to inhibit the KCs after 200 spikes
GGN_PARAMS = {"Vthresh": 200.0}

KC_MBON_TAU_SYN = 3.0
KC_MBON_PARAMS = {"tau": 15.0,
                  "rho": 0.01,
                  "eta": 0.00002,
                  "wMin": 0.0,
                  "wMax": 0.0233}

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
# Current source model, allowing current to be injected into neuron from variable
cs_model = create_current_source_model(
    "cs_model",
    vars=[("magnitude", "scalar")],
    injection_code="injectCurrent(magnitude);")

# Minimal integrate and fire neuron model
if_model = create_neuron_model(
    "IF",
    params=["Vthresh"],
    vars=[("V", "scalar")],
    sim_code=
    """
    V += Isyn;
    """,
    threshold_condition_code=
    """
    V >= Vthresh
    """,
    reset_code=
    """
    V= 0.0;
    """)

# Symmetric STDP learning rule
symmetric_stdp = create_weight_update_model(
    "symmetric_stdp",
    params=["tau", "rho", "eta", "wMin", "wMax"],
    vars=[("g", "scalar")],
    pre_spike_syn_code=
    """
    const scalar dt = t - st_post;
    const scalar timing = exp(-dt / tau) - rho;
    const scalar newWeight = g + (eta * timing);
    g = fmin(wMax, fmax(wMin, newWeight));
    """,
    post_spike_syn_code=
    """
    const scalar dt = t - st_pre;
    const scalar timing = fmax(exp(-dt / tau) - rho, -0.1 * rho);
    const scalar newWeight = g + (eta * timing);
    g = fmin(wMax, fmax(wMin, newWeight));
    """)

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--test", action="store_true", help="Load saved weights (rather than training)")
parser.add_argument("--plot-weight-distribution", action="store_true", help="Plot weight distribution after training")

args = parser.parse_args()

# Reshape and normalise  data
images = mnist.test_images() if args.test else mnist.train_images()
images = np.reshape(images, (images.shape[0], -1)).astype(np.float32)
images /= np.sum(images, axis=1)[:, np.newaxis]
labels = mnist.test_labels() if args.test else mnist.train_labels()

# Create model
model = GeNNModel("float", "mnist_mb")
model.dt = DT

# Create neuron populations
lif_init = {"V": PN_PARAMS["Vreset"], "RefracTime": 0.0}
if_init = {"V": 0.0}
pn = model.add_neuron_population("pn", NUM_PN, "LIF", PN_PARAMS, lif_init)
kc = model.add_neuron_population("kc", NUM_KC, "LIF", LIF_PARAMS, lif_init)
ggn = model.add_neuron_population("ggn", 1, if_model, GGN_PARAMS, if_init)
mbon = model.add_neuron_population("mbon", NUM_MBON, "LIF", LIF_PARAMS, lif_init)

# Turn on spike recording
pn.spike_recording_enabled = True
kc.spike_recording_enabled = True
mbon.spike_recording_enabled = True

# Create current sources to deliver input to network
pn_input = model.add_current_source("pn_input", cs_model, pn , {}, {"magnitude": 0.0})

# Create current sources to deliver input and supervision to network
if not args.test:
    mbon_input = model.add_current_source("mbon_input", cs_model, mbon , {}, {"magnitude": 0.0})

# Create synapse populations
pn_kc_connectivity = None if args.test else init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": PN_KC_FAN_IN})
pn_kc = model.add_synapse_population("pn_kc", "SPARSE",
                                     pn, kc,
                                     init_weight_update("StaticPulseConstantWeight", {"g": PN_KC_WEIGHT}),
                                     init_postsynaptic("ExpCurr", {"tau": PN_KC_TAU_SYN}),
                                     pn_kc_connectivity)

# Load saved connectivity if testing
if args.test:
    pn_kc_ind = np.load("pn_kc_ind.npy")
    pn_kc.set_sparse_connections(pn_kc_ind[0], pn_kc_ind[1])

kc_ggn = model.add_synapse_population("kc_ggn", "DENSE",
                                      kc, ggn,
                                      init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
                                      init_postsynaptic("DeltaCurr"))

ggn_kc = model.add_synapse_population("ggn_kc", "DENSE",
                                      ggn, kc,
                                      init_weight_update("StaticPulseConstantWeight", {"g": -5.0}),
                                      init_postsynaptic("ExpCurr", {"tau": 5.0}))

kc_mbon_weight_update = (init_weight_update("StaticPulse", {}, {"g": np.load("kc_mbon_g.npy")}) if args.test 
                         else init_weight_update(symmetric_stdp, KC_MBON_PARAMS, {"g": 0.0}))
kc_mbon = model.add_synapse_population("kc_mbon", "DENSE",
                                       kc, mbon,
                                       kc_mbon_weight_update,
                                       init_postsynaptic("ExpCurr", {"tau": KC_MBON_TAU_SYN}))
                                       
# Convert present time into timesteps
present_timesteps = int(round(PRESENT_TIME_MS / DT))

# Build model and load it
model.build()
model.load(num_recording_timesteps=present_timesteps)

def reset_spike_times(pop):
    pop.spike_times.view[:] = -np.finfo(np.float32).max
    pop.spike_times.push_to_device()

def reset_out_post(pop):
    pop.out_post.view[:] = 0.0
    pop.out_post.push_to_device()

def reset_neuron(pop, var_init):
    # Reset variables
    for var_name, var_val in var_init.items():
        var = pop.vars[var_name]

        # Reset to initial value and push to device
        var.view[:] = var_val
        var.push_to_device()

# Present images
num_correct = 0
for s in tqdm(range(images.shape[0])):
    # Set training image
    pn_input.vars["magnitude"].view[:] = images[s] * INPUT_SCALE
    pn_input.vars["magnitude"].push_to_device()

    # Turn on correct output neuron
    if not args.test:
        mbon_input.vars["magnitude"].view[:] = 0
        mbon_input.vars["magnitude"].view[labels[s]] = MBON_STIMULUS_CURRENT
        mbon_input.vars["magnitude"].push_to_device()

    # Simulate present timesteps
    for i in range(present_timesteps):
        model.step_time()

    # Reset neuron state
    reset_neuron(pn, lif_init)
    reset_neuron(kc, lif_init)
    reset_neuron(ggn, if_init)
    reset_neuron(mbon, lif_init)

    # Reset spike times
    if not args.test:
        reset_spike_times(kc)
        reset_spike_times(mbon)

    # Reset synapse state
    reset_out_post(pn_kc)
    reset_out_post(ggn_kc)
    reset_out_post(kc_mbon)
    
    if args.test:
         # Download spikes from GPU
        model.pull_recording_buffers_from_device();

        # Determine the classification and count correct
        mbon_spike_times, mbon_spike_ids = mbon.spike_recording_data[0]
        if len(mbon_spike_times) > 0:
            if mbon_spike_ids[np.argmin(mbon_spike_times)] == labels[s]:
                num_correct += 1

if args.test:
    print(f"\n{num_correct}/{images.shape[0]} correct ({(num_correct * 100.0) / images.shape[0]} %%)")
else:
    pn_kc.pull_connectivity_from_device()
    kc_mbon.vars["g"].pull_from_device()
    
    # Save weighs and connectivity
    kc_mbon_g_view = kc_mbon.vars["g"].view
    np.save("kc_mbon_g.npy", kc_mbon_g_view)
    np.save("pn_kc_ind.npy", np.vstack((pn_kc.get_sparse_pre_inds(),
                                        pn_kc.get_sparse_post_inds())))

    # Plot weight distribution
    if args.plot_weight_distribution:
        from matplotlib import pyplot as plt

        fig, axis = plt.subplots(figsize=(10, 5))
        axis.hist(kc_mbon_g_view, bins=100)
        axis.axvline(np.average(kc_mbon_g_view), linestyle="--")
        axis.set_xlabel("Weight [nA]")
        axis.set_ylabel("Count");
        plt.show()
