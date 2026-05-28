import numpy as np
import pytest
from pygenn import types
from scipy import stats

from pygenn import VarAccess, VarAccessMode
from pygenn import (create_neuron_model, create_var_ref,
                    create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_toeplitz_connectivity,
                    init_weight_update, init_var)

# Weight update models which copy a PRESYNAPTIC neuron variables
# into synapse during various kinds of synaptic event
pre_learn_post_weight_update_model = create_weight_update_model(
    "pre_learn_post_weight_update",
    vars=[("w", "scalar")],
    pre_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],
    
    post_spike_syn_code=
    """
    w = s;
    """)

pre_sim_weight_update_model = create_weight_update_model(
    "pre_sim_weight_update",
    vars=[("w", "scalar")],
    pre_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    pre_spike_syn_code=
    """
    w = s;
    """)

pre_event_weight_update_model = create_weight_update_model(
    "pre_sim_weight_update",
    vars=[("w", "scalar")],
    pre_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    pre_event_syn_code=
    """
    w = s;
    """,
    pre_event_threshold_condition_code=
    """
    t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
    """)

# Weight update models which copy a POSTSYNAPTIC neuron variables
# into synapse during various kinds of synaptic event
post_sim_weight_update_model = create_weight_update_model(
    "post_sim_weight_update",
    vars=[("w", "scalar")],
    post_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    pre_spike_syn_code=
    """
    w = s;
    """)

post_learn_post_weight_update_model = create_weight_update_model(
    "post_learn_post_weight_update",
    vars=[("w", "scalar")],
    post_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    post_spike_syn_code=
    """
    w = s;
    """)

post_event_weight_update_model = create_weight_update_model(
    "pre_sim_weight_update",
    vars=[("w", "scalar")],
    post_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    pre_event_syn_code=
    """
    w = s;
    """,
    pre_event_threshold_condition_code=
    """
    t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
    """)

post_neuron_model = create_neuron_model(
    "post_neuron",
    sim_code=
    """
    x = Isyn;
    """,
    vars=[("x", "scalar")])
 
spike_source_array_extra_var_neuron_model = create_neuron_model(
    "spike_source_array_extra_var",
    threshold_condition_code=
    """
    startSpike != endSpike && t >= spikeTimes[startSpike]
    """,
    reset_code=
    """
    startSpike++;
    """,
    vars=[("constant", "scalar"),
          ("startSpike", "unsigned int"), 
          ("endSpike", "unsigned int", VarAccess.READ_ONLY_DUPLICATE)],
     extra_global_params=[("spikeTimes", "scalar*")])

scaled_weight_update_model = create_weight_update_model(
        "scaled_weight_update",
        vars=[("g", "scalar", VarAccess.READ_ONLY)],
        pre_neuron_var_refs=[("x", "scalar", VarAccessMode.READ_ONLY)],
        pre_spike_syn_code=
        """
        addToPost(g * x);
        """)
@pytest.mark.parametrize("precision", [types.Double, types.Float])
@pytest.mark.parametrize("delay", [0, 20])
def test_pre_post_neuron_var(make_model, backend, precision, delay):
    # Neuron model which fires at t = id ms and every 10 ms after that
    pattern_spike_neuron_model = create_neuron_model(
        "pattern_spike_neuron",
        sim_code=
        """
        s = t + shift;
        """,
        threshold_condition_code=
        """
        t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
        """,
        vars=[("s", "scalar"), ("shift", "scalar", VarAccess.READ_ONLY)])

    # Neuron model which updates a state variable with time + per-neuron shift
    shift_pattern_neuron_model = create_neuron_model(
        "shift_pattern_neuron",
        sim_code=
        """
        s = t + shift;
        """,
        vars=[("s", "scalar"), ("shift", "scalar", VarAccess.READ_ONLY)])


    model = make_model(precision, "test_pre_post_neuron_var", backend=backend)
    model.dt = 1.0

    # Create pre and postsynaptic neuron populations
    float_min = np.finfo(np.float32).min
    shift = np.arange(0.0, 100.0, 10.0)
    pre_n_pop = model.add_neuron_population("PreNeurons", 10, pattern_spike_neuron_model, 
                                            {}, {"s": float_min, "shift": shift})
    post_n_pop = model.add_neuron_population("PostNeurons", 10, shift_pattern_neuron_model, 
                                             {}, {"s": float_min, "shift": shift})

    # Add synapse models testing various ways of reading presynaptic WU vars
    s_pre_learn_post_sparse_pop = model.add_synapse_population(
        "PreLearnPostSparseSynapses", "SPARSE",
        post_n_pop, pre_n_pop,
        init_weight_update(pre_learn_post_weight_update_model, {}, {"w": float_min},
                           pre_var_refs={"s": "s"}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    s_pre_learn_post_sparse_pop.axonal_delay_steps = delay

    s_pre_sim_sparse_pop = model.add_synapse_population(
        "PreSimSparseSynapses", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(pre_sim_weight_update_model, {}, {"w": float_min},
                           pre_var_refs={"s": create_var_ref(pre_n_pop, "s")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    s_pre_sim_sparse_pop.axonal_delay_steps = delay

    s_pre_event_sparse_pop = model.add_synapse_population(
        "PreEventSparseSynapses", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(pre_event_weight_update_model, {}, {"w": float_min},
                           pre_var_refs={"s": "s"}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    s_pre_event_sparse_pop.axonal_delay_steps = delay
        
    # Add synapse models testing various ways of reading post WU vars
    s_post_learn_post_sparse_pop = model.add_synapse_population(
        "PostLearnPostSparseSynapses", "SPARSE",
        post_n_pop, pre_n_pop,
        init_weight_update(post_learn_post_weight_update_model, {}, {"w": float_min},
                           post_var_refs={"s": create_var_ref(pre_n_pop, "s")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    s_post_learn_post_sparse_pop.back_prop_delay_steps = delay
    s_post_sim_sparse_pop = model.add_synapse_population(
        "PostSimSparseSynapses", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(post_sim_weight_update_model, {}, {"w": float_min},
                           post_var_refs={"s": "s"}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    s_post_sim_sparse_pop.back_prop_delay_steps = delay
    
    s_post_event_sparse_pop = model.add_synapse_population(
        "PostEventSparseSynapses", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(post_event_weight_update_model, {}, {"w": float_min},
                           post_var_refs={"s": create_var_ref(post_n_pop, "s")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    s_post_event_sparse_pop.back_prop_delay_steps = delay

    # Build model and load
    model.build()
    model.load()

    # Pull all synaptic connectivity from device
    synapse_groups = [s_post_sim_sparse_pop, 
                      s_post_learn_post_sparse_pop,
                      s_post_event_sparse_pop,
                      s_pre_sim_sparse_pop, 
                      s_pre_learn_post_sparse_pop,
                      s_pre_event_sparse_pop]
    for s in synapse_groups:
        s.pull_connectivity_from_device()
    
    while model.timestep < 100:
        model.step_time()
        
        # Calculate time of spikes we SHOULD be reading
        # **NOTE** we delay by delay + 2 timesteps because:
        # 1) delay = delay
        # 2) spike times are read in synapse kernel one timestep AFTER being emitted
        # 3) t is incremented one timestep at te end of StepGeNN
        delayed_time = (11 * np.arange(10)) + (10.0 * np.floor((model.t - delay - 2.0 - np.arange(10)) / 10.0))
        delayed_time[delayed_time < (10 * np.arange(10))] = float_min
        
        # Loop through synapse groups and compare value of w with delayed time
        for s in synapse_groups:
            s.vars["w"].pull_from_device()
            w_value = s.vars["w"].values
            if not np.allclose(delayed_time, w_value):
                assert False, f"{s.name} var has wrong value ({w_value} rather than {delayed_time})"


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_pre_neuron_var_kernel(make_model, backend, precision):
    model = make_model(precision, "test_pre_neuron_var_kernel", backend=backend)
    model.dt = 1.0


    # (Normalised) horizontal Sobel convolution kernel
    vertical_kernel = np.asarray([[1.0,   0.0,    -1.0],
                                [2.0,   0.0,    -2.0],
                                [1.0,   0.0,    -1.0]])

    # (Normalised) vertical Sobel convolution kernel
    horizontal_kernel = np.asarray([[1.0,     2.0,    1.0],
                                    [0.0,     0.0,    0.0],
                                    [-1.0,    -2.0,   -1.0]])

    # Create spike source array to present test pattern
    test_pattern = np.load("test_pattern.npy")
    end_spikes = np.cumsum(np.bincount(test_pattern, minlength=64 * 64))
    start_spikes = np.concatenate(([0,], end_spikes[:-1]))
    pre_pop = model.add_neuron_population("SpikeSource", 64 * 64, spike_source_array_extra_var_neuron_model,
                                          {}, {"startSpike": start_spikes, "endSpike": end_spikes, "constant": 1.0})
    pre_pop.extra_global_params["spikeTimes"].set_init_values(np.zeros_like(test_pattern))

    # Add postsynaptic populations to receive horizontal and vertical edges
    post_toeplitz_horiz_pop = model.add_neuron_population(
        "PostHorizNeurons", 62 * 62, post_neuron_model, 
        {}, {"x": 0.0})

    post_toeplitz_vert_pop = model.add_neuron_population(
        "PostVertNeurons", 62 * 62, post_neuron_model, 
        {}, {"x": 0.0})
    
    # Add convolutional toeplitz connectivity
    conv_toeplitz_params = {"conv_kh": 3, "conv_kw": 3,
                            "conv_ih": 64, "conv_iw": 64, "conv_ic": 1,
                            "conv_oh": 62, "conv_ow": 62, "conv_oc": 1}
    model.add_synapse_population(
        "ToeplitzHorizSynapse", "TOEPLITZ",
        pre_pop, post_toeplitz_horiz_pop,
 
        init_weight_update(scaled_weight_update_model, {}, {"g": horizontal_kernel.flatten()},
                           pre_var_refs={"x": create_var_ref(pre_pop, "constant")}),
        init_postsynaptic("DeltaCurr"),
        init_toeplitz_connectivity("Conv2D", conv_toeplitz_params))
    model.add_synapse_population(
        "ToeplitzVertSynapse", "TOEPLITZ",
        pre_pop, post_toeplitz_vert_pop,
 
        init_weight_update(scaled_weight_update_model, {}, {"g": vertical_kernel.flatten()},
                           pre_var_refs={"x": create_var_ref(pre_pop, "constant")}),
        init_postsynaptic("DeltaCurr"),
        init_toeplitz_connectivity("Conv2D", conv_toeplitz_params))

    # Build model and load
    model.build()
    model.load()
    
    # Step time twice - in first timestep spikes will be emitted 
    # by pre_pop. In second, they will be received by the post_pops
    model.step_time()
    model.step_time()
    
    # Download output variables from device
    post_toeplitz_horiz_pop.vars["x"].pull_from_device()
    post_toeplitz_vert_pop.vars["x"].pull_from_device()

    # Check against correct convolutions
    correct_horizontal = np.load("horizontal_output.npy") 
    correct_vertical = np.load("vertical_output.npy")
    assert np.allclose(post_toeplitz_horiz_pop.vars["x"].view, 
                       correct_horizontal)
    assert np.allclose(post_toeplitz_vert_pop.vars["x"].view, 
                       correct_vertical)
