"""
PyGeNN implementation of SuperSpike
===================================
This example model is a reimplementation of the model developed by 
Friedemann Zenke and Surya Ganguli [Zenke and Ganguli, 2018]_. It uses the SuperSpike 
learning rule to learn the transformation between fixed spike trains of 
Poisson noise and a target spiking output (by default the Radcliffe Camera at Oxford).

This example can be used as follows:

.. argparse::
   :filename: ../userproject/superspike_demo.py
   :func: get_parser
   :prog: superspike_demo
"""
import numpy as np

from argparse import ArgumentParser
from pygenn import (create_custom_update_model, create_neuron_model,
                    create_postsynaptic_model, create_var_ref,
                    create_weight_update_model, create_wu_var_ref,
                    init_postsynaptic, init_var, init_weight_update)
from pygenn import GeNNModel

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
TIMESTEP_MS = 0.1

# Network structure
NUM_INPUT = 200
NUM_OUTPUT = 200
NUM_HIDDEN = 256

# Model parameters
TAU_RISE_MS = 5.0
TAU_DECAY_MS = 10.0
TAU_RMS_MS = 30000.0
TAU_AVG_ERR_MS = 10000.0
R0 = 0.001 * 1000.0
EPSILON = 1E-32
TAU_DECAY_S = TAU_DECAY_MS / 1000.0
TAU_RISE_S = TAU_RISE_MS / 1000.0
TAU_AVG_ERR_S = TAU_AVG_ERR_MS / 1000.0
SCALE_TR_ERR_FLT = 1.0 / (pow((TAU_DECAY_S * TAU_RISE_S)/(TAU_DECAY_S - TAU_RISE_S),2) * (TAU_DECAY_S/2+TAU_RISE_S/2-2*(TAU_DECAY_S*TAU_RISE_S)/(TAU_DECAY_S+TAU_RISE_S))) / TAU_AVG_ERR_S

# Weights
# **NOTE** Auryn units are volts, seconds etc so essentially 1000x GeNN parameters
W_MIN = -0.1 * 1000.0
W_MAX = 0.1 * 1000.0
W0 = 0.05 * 1000.0

# Experiment parameters
INPUT_FREQ_HZ = 5.0
UPDATE_TIME_MS = 500.0
TRIAL_MS = 1890.0

# Convert parameters to timesteps
UPDATE_TIMESTEPS = int(UPDATE_TIME_MS / TIMESTEP_MS)
TRIAL_TIMESTEPS = int(TRIAL_MS / TIMESTEP_MS)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def calc_t_peak(tau_rise, tau_decay):
    return ((tau_decay * tau_rise) / (tau_decay - tau_rise)) * np.log(tau_decay / tau_rise)

def write_spike_file(filename, data):
    np.savetxt(filename, np.column_stack(data[0]), fmt=["%f","%d"],
               delimiter=",", header="Time [ms], Neuron ID")

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
r_max_prop_model = create_custom_update_model(
    "r_max_prop",
    params=["updateTime", "tauRMS", "epsilon", "wMin", "wMax", "r0"],
    vars=[("upsilon", "scalar")],
    derived_params=[("updateTimesteps", lambda pars, dt: pars["updateTime"] / dt),
                    ("expRMS", lambda pars, dt: np.exp(-pars["updateTime"] / pars["tauRMS"]))],
    var_refs=[("m", "scalar"), ("variable", "scalar")],
    update_code="""
    // Get gradients
    const scalar gradient = m / updateTimesteps;
    // Calculate learning rate r
    upsilon = fmax(upsilon * expRMS, gradient * gradient);
    const scalar r = r0 / (sqrt(upsilon) + epsilon);
    // Update synaptic parameter
    variable += r * gradient;
    variable = fmin(wMax, fmax(wMin, variable));
    m = 0.0;
    """)

superspike_model = create_weight_update_model(
    "superspike",
    params=["tauRise", "tauDecay", "beta", "Vthresh"],
    vars=[("w", "scalar"), ("e", "scalar"), 
          ("lambda", "scalar"), ("m", "scalar")],
    pre_vars=[("z", "scalar"), ("zTilda", "scalar")],
    post_vars=[("sigmaPrime", "scalar")],
    post_neuron_var_refs=[("V", "scalar"), ("errTilda", "scalar")],

    pre_spike_syn_code="""
    addToPost(w);
    """,

    pre_spike_code="""
    z += 1.0;
    """,
    pre_dynamics_code="""
    // filtered presynaptic trace
    z += (-z / tauRise) * dt;
    zTilda += ((-zTilda + z) / tauDecay) * dt;
    """,

    post_dynamics_code="""
    // filtered partial derivative
    if(V < -80.0) {
       sigmaPrime = 0.0;
    }
    else {
       const scalar onePlusHi = 1.0 + fabs(beta * 0.001 * (V - Vthresh));
       sigmaPrime = beta / (onePlusHi * onePlusHi);
    }
    """,

    synapse_dynamics_code="""
    // Filtered eligibility trace
    e += (zTilda * sigmaPrime - e / tauRise) * dt;
    lambda += ((-lambda + e) / tauDecay) * dt;

    // Get error from neuron model and compute full
    // expression under integral and calculate m
    m += lambda * errTilda;
    """)

feedback_model = create_weight_update_model(
    "feedback",
    vars=[("w", "scalar")],
    pre_neuron_var_refs=[("errTilda", "scalar")],
    synapse_dynamics_code="""
    addToPost(w * errTilda);
    """)

hidden_neuron_model = create_neuron_model(
    "hidden",
    params=["C", "tauMem", "Vrest", "Vthresh", "tauRefrac"],
    vars=[("V", "scalar"), ("refracTime", "scalar"), ("errTilda", "scalar")],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)],
    derived_params=[("ExpTC", lambda pars, dt: np.exp(-dt / pars["tauMem"])),
                    ("Rmembrane", lambda pars, dt: pars["tauMem"] / pars["C"])],
   
    sim_code="""
    // membrane potential dynamics
    if (refracTime == tauRefrac) {
        V = Vrest;
    }
    if (refracTime <= 0.0) {
        scalar alpha = (Isyn * Rmembrane) + Vrest;
        V = alpha - (ExpTC * (alpha - V));
    }
    else {
        refracTime -= dt;
    }
    // error
    errTilda = ISynFeedback;
    """,
    reset_code="""
    refracTime = tauRefrac;
    """,
    threshold_condition_code="""
    refracTime <= 0.0 && V >= Vthresh
    """)

output_neuron_model = create_neuron_model(
    "output",
    params=["C", "tauMem", "Vrest", "Vthresh", "tauRefrac",
            "tauRise", "tauDecay", "tauAvgErr"],
    vars=[("V", "scalar"), ("refracTime", "scalar"), ("errRise", "scalar"),
          ("errTilda", "scalar"), ("avgSqrErr", "scalar"), ("errDecay", "scalar"),
           ("startSpike", "unsigned int"), ("endSpike", "unsigned int")],
    extra_global_params=[("spikeTimes", "scalar*")],
    derived_params=[("ExpTC", lambda pars, dt: np.exp(-dt / pars["tauMem"])),
                    ("Rmembrane", lambda pars, dt: pars["tauMem"] / pars["C"]),
                    ("normFactor", lambda pars, dt: 1.0 / (-np.exp(-calc_t_peak(pars["tauRise"], pars["tauDecay"]) / pars["tauRise"]) + np.exp(-calc_t_peak(pars["tauRise"], pars["tauDecay"]) / pars["tauDecay"]))),
                    ("tRiseMult", lambda pars, dt: np.exp(-dt / pars["tauRise"])),
                    ("tDecayMult", lambda pars, dt: np.exp(-dt / pars["tauDecay"])),
                    ("tPeak", lambda pars, dt: calc_t_peak(pars["tauRise"], pars["tauDecay"])),
                    ("mulAvgErr", lambda pars, dt: np.exp(-dt / pars["tauAvgErr"]))],

    sim_code="""
    // membrane potential dynamics
    if (refracTime == tauRefrac) {
        V = Vrest;
    }
    if (refracTime <= 0.0) {
        scalar alpha = (Isyn * Rmembrane) + Vrest;
        V = alpha - (ExpTC * (alpha - V));
    }
    else {
        refracTime -= dt;
    }
    // error
    scalar sPred = 0.0;
    if (startSpike != endSpike && t >= spikeTimes[startSpike]) {
        startSpike++;
        sPred = 1.0;
    }
    const scalar sReal = (refracTime <= 0.0 && V >= Vthresh) ? 1.0 : 0.0;
    const scalar mismatch = sPred - sReal;
    errRise = (errRise * tRiseMult) + mismatch;
    errDecay = (errDecay * tDecayMult) + mismatch;
    errTilda = (errDecay - errRise) * normFactor;
    // calculate average error trace
    const scalar temp = errTilda * errTilda * dt * 0.001;
    avgSqrErr *= mulAvgErr;
    avgSqrErr += temp;
    """,
    reset_code="""
    refracTime = tauRefrac;
    """,
    threshold_condition_code="""
    refracTime <= 0.0 && V >= Vthresh
    """)

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--record-trial", type=int, nargs="*", required=True, help="Index of trial(s) to record")
    parser.add_argument("--target-file", type=str, default="oxford-target.ras", help="Filename of spike file to train model on")
    parser.add_argument("--num-trials", type=int, default=600, help="Number of trials to train for")
    parser.add_argument("--kernel-profiling", action="store_true", help="Output kernel profiling data")
    parser.add_argument("--save-data", action="store_true", help="Save spike data (rather than plotting it)")
    return parser


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_parser().parse_args()
    # Sort trial indices to record
    args.record_trial = sorted(args.record_trial)

    # ----------------------------------------------------------------------------
    # Load target data
    # ----------------------------------------------------------------------------
    # Load target data
    target_spikes = np.loadtxt(args.target_file,
                               dtype={"names": ("time", "neuron_id"),
                                      "formats": (float, int)})

    # Make neuron IDs zero-based
    target_spikes["neuron_id"] -= 1

    # Convert times to milliseconds
    target_spikes["time"] *= 1000.0

    # Sort first by neuron id and then by time
    target_spikes = np.sort(target_spikes, order=["neuron_id", "time"])

    # Count number of spikes
    target_neuron_end_times = np.cumsum(np.bincount(target_spikes["neuron_id"], minlength=NUM_OUTPUT))
    target_neuron_start_times = np.concatenate(([0], target_neuron_end_times[:-1]))

    # ----------------------------------------------------------------------------
    # Generate frozen poisson input
    # ----------------------------------------------------------------------------
    input_isi_ms = 1000.0 / INPUT_FREQ_HZ

    # Generate time of first spike for each neuron
    input_spike_times = input_isi_ms * np.random.exponential(size=NUM_INPUT)
    input_spike_times = np.reshape(input_spike_times, (1, NUM_INPUT))

    while True:
        # Generate vector of spike times
        s = input_isi_ms * np.random.exponential(size=NUM_INPUT)

        # Add previous times
        s += input_spike_times[-1,:]

        # If all neurons have reached end of trial
        if np.all(s >= TRIAL_MS):
            break
        # Otherwise stack
        else:
            input_spike_times = np.vstack((input_spike_times, s))

    # Count spikes per input neuron
    input_spikes_per_neuron = np.sum(input_spike_times < TRIAL_MS, axis=0)

    # Concatenate spikes within trial together
    input_spikes = np.concatenate([input_spike_times[:input_spikes_per_neuron[i],i] 
                                   for i in range(NUM_INPUT)])

    # Calculate indices
    input_neuron_end_times = np.cumsum(input_spikes_per_neuron)
    input_neuron_start_times = np.concatenate(([0], input_neuron_end_times[:-1]))

    # ----------------------------------------------------------------------------
    # Neuron initialisation
    # ----------------------------------------------------------------------------
    input_init_vars = {"startSpike": input_neuron_start_times, "endSpike": input_neuron_end_times}

    hidden_params = {"C" : 10.0, "tauMem": 10.0, "Vrest": -60.0, 
                     "Vthresh": -50.0 , "tauRefrac": 5.0}
    hidden_init_vars = {"V": -60.0, "refracTime": 0.0, "errTilda": 0.0}

    output_params = {"C": 10.0, "tauMem": 10.0, "Vrest": -60.0, 
                     "Vthresh": -50.0, "tauRefrac": 5.0, "tauRise": TAU_RISE_MS, 
                     "tauDecay": TAU_DECAY_MS, "tauAvgErr": TAU_AVG_ERR_MS}
    output_init_vars = {"V": -60.0, "refracTime": 0.0, "errRise": 0.0, 
                        "errTilda": 0.0, "errDecay": 0.0, "avgSqrErr": 0.0,
                        "startSpike": target_neuron_start_times, "endSpike": target_neuron_end_times}

    # ----------------------------------------------------------------------------
    # Synapse initialisation
    # ----------------------------------------------------------------------------
    superspike_params = {"tauRise": TAU_RISE_MS, "tauDecay": TAU_DECAY_MS, "beta": 1000.0, "Vthresh": -50.0}
    superspike_pre_init_vars = {"z": 0.0, "zTilda": 0.0}
    superspike_post_init_vars = {"sigmaPrime": 0.0}

    input_hidden_weight_dist_params = {"mean": 0.0, "sd": W0 / np.sqrt(float(NUM_INPUT)),
                                       "min": W_MIN, "max": W_MAX}
    input_hidden_init_vars = {"w": init_var("NormalClipped", input_hidden_weight_dist_params),
                              "e": 0.0, "lambda": 0.0, "m": 0.0}

    hidden_output_weight_dist_params = {"mean": 0.0, "sd": W0 / np.sqrt(float(NUM_HIDDEN)),
                                        "min": W_MIN, "max": W_MAX}
    hidden_output_init_vars = {"w": init_var("NormalClipped", hidden_output_weight_dist_params),
                               "e": 0.0, "lambda": 0.0, "m": 0.0}     
           
    # ----------------------------------------------------------------------------
    # Custom update initialisation
    # ----------------------------------------------------------------------------
    r_max_prop_params = {"updateTime": UPDATE_TIME_MS, "tauRMS": TAU_RMS_MS, 
                         "epsilon": EPSILON, "wMin": W_MIN, "wMax": W_MAX, "r0": R0}

    # ----------------------------------------------------------------------------
    # Model description
    # ----------------------------------------------------------------------------
    model = GeNNModel("float", "superspike_demo", generateLineInfo=True)
    model.dt = TIMESTEP_MS
    model.timing_enabled = args.kernel_profiling

    # Add neuron populations
    input = model.add_neuron_population("Input", NUM_INPUT, "SpikeSourceArray", 
                                        {}, input_init_vars)
    hidden = model.add_neuron_population("Hidden", NUM_HIDDEN, hidden_neuron_model, 
                                         hidden_params, hidden_init_vars)
    output = model.add_neuron_population("Output", NUM_OUTPUT, output_neuron_model, 
                                         output_params, output_init_vars)

    input.extra_global_params["spikeTimes"].set_init_values(input_spikes)
    output.extra_global_params["spikeTimes"].set_init_values(target_spikes["time"])

    # Turn on recording
    any_recording = (len(args.record_trial) > 0)
    input.spike_recording_enabled = any_recording
    hidden.spike_recording_enabled = any_recording
    output.spike_recording_enabled = any_recording

    # Add synapse populations
    input_hidden = model.add_synapse_population(
        "InputHidden", "DENSE",
        input, hidden,
        init_weight_update(superspike_model, superspike_params, input_hidden_init_vars, superspike_pre_init_vars, superspike_post_init_vars,
                           post_var_refs={"V": create_var_ref(hidden, "V"), "errTilda": create_var_ref(hidden, "errTilda")}),
        init_postsynaptic("ExpCurr", {"tau": 5.0}))

    hidden_output = model.add_synapse_population(
        "HiddenOutput", "DENSE",
        hidden, output,
        init_weight_update(superspike_model, superspike_params, hidden_output_init_vars, superspike_pre_init_vars, superspike_post_init_vars,
                           post_var_refs={"V": create_var_ref(output, "V"), "errTilda": create_var_ref(output, "errTilda")}),
        init_postsynaptic("ExpCurr", {"tau": 5.0}))

    output_hidden = model.add_synapse_population(
        "OutputHidden", "DENSE",
        output, hidden,
        init_weight_update(feedback_model, {}, {"w": 0.0}, pre_var_refs={"errTilda": create_var_ref(output, "errTilda")}),
        init_postsynaptic("DeltaCurr"))
    output_hidden.post_target_var = "ISynFeedback"

    # Add custom update for calculating initial tranpose weights
    model.add_custom_update("input_hidden_transpose", "CalculateTranspose", "Transpose",
                            {}, {}, {"variable": create_wu_var_ref(hidden_output, "w", output_hidden, "w")})

    # Add custom updates for gradient update
    input_hidden_optimiser_var_refs = {"m": create_wu_var_ref(input_hidden, "m"), 
                                       "variable": create_wu_var_ref(input_hidden, "w")}
    input_hidden_optimiser = model.add_custom_update("input_hidden_optimiser", "GradientLearn", r_max_prop_model,
                                                     r_max_prop_params, {"upsilon": 0.0}, input_hidden_optimiser_var_refs)
    input_hidden_optimiser.set_param_dynamic("r0")

    hidden_output_optimiser_var_refs = {"m": create_wu_var_ref(hidden_output, "m"), 
                                       "variable": create_wu_var_ref(hidden_output, "w", output_hidden, "w")}
    hidden_output_optimiser = model.add_custom_update("hidden_output_optimiser", "GradientLearn", r_max_prop_model,
                                                      r_max_prop_params, {"upsilon": 0.0}, hidden_output_optimiser_var_refs)
    hidden_output_optimiser.set_param_dynamic("r0")

    # Build and load model
    model.build()
    model.load(num_recording_timesteps=TRIAL_TIMESTEPS)

    # Calculate initial transpose feedback weights
    model.custom_update("CalculateTranspose")

    # Loop through trials
    output_avg_sqr_err_var = output.vars["avgSqrErr"]
    current_r0 = R0
    timestep = 0
    input_spikes = []
    hidden_spikes = []
    output_spikes = []
    for trial in range(args.num_trials):
        # Reduce learning rate every 400 trials
        if trial != 0 and (trial % 400) == 0:
            current_r0 *= 0.1

            input_hidden_optimiser.set_dynamic_param_value("r0", current_r0)
            hidden_output_optimiser.set_dynamic_param_value("r0", current_r0)

        # Display trial number peridically
        if trial != 0 and (trial % 10) == 0:
            # Get average square error
            output_avg_sqr_err_var.pull_from_device()

            # Calculate mean error
            time_s = timestep * TIMESTEP_MS / 1000.0;
            mean_error = np.sum(output_avg_sqr_err_var.view) / float(NUM_OUTPUT);
            mean_error *= SCALE_TR_ERR_FLT / (1.0 - np.exp(-time_s / TAU_AVG_ERR_S) + 1.0E-9);

            print("Trial %u (r0 = %f, error = %f)" % (trial, current_r0, mean_error))

        # Reset model timestep
        model.timestep = 0

        # Loop through timesteps within trial
        for i in range(TRIAL_TIMESTEPS):
            model.step_time()

            # If it's time to update weights
            if timestep != 0 and (timestep % UPDATE_TIMESTEPS) == 0:
                model.custom_update("GradientLearn");
            timestep+=1;


        # Reset spike sources by re-uploading starting spike indices
        # **TODO** build repeating spike source array
        input.vars["startSpike"].push_to_device()
        output.vars["startSpike"].push_to_device()

        if trial in args.record_trial:
            model.pull_recording_buffers_from_device();
            
            if args.save_data:
                write_spike_file("input_spikes_%u.csv" % trial, input.spike_recording_data)
                write_spike_file("hidden_spikes_%u.csv" % trial, hidden.spike_recording_data)
                write_spike_file("output_spikes_%u.csv" % trial, output.spike_recording_data)
            else:
                input_spikes.append(input.spike_recording_data[0])
                hidden_spikes.append(hidden.spike_recording_data[0])
                output_spikes.append(output.spike_recording_data[0])

    if args.kernel_profiling:
        print("Init: %f" % model.init_time)
        print("Init sparse: %f" % model.init_sparse_time)
        print("Neuron update: %f" % model.neuron_update_time)
        print("Presynaptic update: %f" % model.presynaptic_update_time)
        print("Synapse dynamics: %f" % model.synapse_dynamics_time)
        print("Gradient learning custom update: %f" % model.get_custom_update_time("GradientLearn"))
        print("Gradient learning custom update transpose: %f" % model.get_custom_update_transpose_time("GradientLearn"))

    if not args.save_data:
        import matplotlib.pyplot as plt
        
        # Create plot
        fig, axes = plt.subplots(3, len(input_spikes), sharex="col", sharey="row")

        for i, spikes in enumerate(zip(input_spikes, hidden_spikes, output_spikes)):
            # Plot spikes
            start_time_s = float(args.record_trial[i]) * 1.890
            axes[0, i].scatter(start_time_s + (spikes[0][0] / 1000.0), spikes[0][1], s=2, edgecolors="none")
            axes[1, i].scatter(start_time_s + (spikes[1][0] / 1000.0), spikes[1][1], s=2, edgecolors="none")
            axes[2, i].scatter(start_time_s + (spikes[2][0] / 1000.0), spikes[2][1], s=2, edgecolors="none")

            axes[2, i].set_xlabel("Time [s]")

        axes[0, 0].set_ylabel("Neuron number")
        axes[1, 0].set_ylabel("Neuron number")
        axes[2, 0].set_ylabel("Neuron number")

        # Show plot
        plt.show()
