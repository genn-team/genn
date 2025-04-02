import numpy as np
import pytest
from pygenn import types
from scipy import stats

from pygenn import VarAccess, VarAccessMode
from pygenn import (create_neuron_model, create_postsynaptic_model,
                    create_var_ref, create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_weight_update, init_var)

# Weight update models which copy a POSTSYNAPTIC neuron variables
# into synapse during various kinds of synaptic event
post_sim_weight_update_model = create_weight_update_model(
    "post_sim_weight_update",
    vars=[("w", "scalar")],
    psm_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    pre_spike_syn_code=
    """
    w = s;
    """)

post_learn_post_weight_update_model = create_weight_update_model(
    "post_learn_post_weight_update",
    vars=[("w", "scalar")],
    psm_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    post_spike_syn_code=
    """
    w = s;
    """)

post_event_weight_update_model = create_weight_update_model(
    "pre_sim_weight_update",
    vars=[("w", "scalar")],
    psm_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    pre_event_syn_code=
    """
    w = s;
    """,
    pre_event_threshold_condition_code=
    """
    t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
    """)

post_sim_delay_weight_update_model = create_weight_update_model(
    "post_sim_delay_weight_update",
    vars=[("w", "scalar")],
    params=[("d", "int")],
    psm_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    pre_spike_syn_code=
    """
    w = s[d];
    """)

post_learn_post_delay_weight_update_model = create_weight_update_model(
    "post_learn_post_delay_weight_update",
    vars=[("w", "scalar")],
    params=[("d", "int")],
    psm_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    post_spike_syn_code=
    """
    w = s[d];
    """)

post_event_delay_weight_update_model = create_weight_update_model(
    "pre_sim_delay_weight_update",
    vars=[("w", "scalar")],
    params=[("d", "int")],
    psm_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    pre_event_syn_code=
    """
    w = s[d];
    """,
    pre_event_threshold_condition_code=
    """
    t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
    """)
    
@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
@pytest.mark.parametrize("delay", [0, 20])
@pytest.mark.parametrize("fuse", [False, True])
def test_post_psm_var(make_model, backend, precision, delay, fuse):
    # Postsynaptic model which updates a state variable with time + per-neuron shift
    pattern_postsynaptic_model = create_postsynaptic_model(
        "pattern_postsynaptic_model",
        sim_code=
        """
        s = t + shift;
        """,
        vars=[("s", "scalar"), ("shift", "scalar", VarAccess.READ_ONLY)])

    # Neuron model which fires at t = id ms and every 10 ms after that
    spike_neuron_model = create_neuron_model(
        "pattern_spike_neuron",
        threshold_condition_code=
        """
        t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
        """)

    # Empty neuron model
    empty_neuron_model = create_neuron_model(
        "empty_neuron")


    model = make_model(precision, "test_post_psm_var", backend=backend)
    model.dt = 1.0
    model.fuse_postsynaptic_models = fuse

    # Create pre and postsynaptic neuron populations
    float_min = np.finfo(np.float32).min
    shift = np.arange(0.0, 100.0, 10.0)
    pre_n_pop = model.add_neuron_population("PreNeurons", 10, spike_neuron_model)
    post_n_pop = model.add_neuron_population("PostNeurons", 10, empty_neuron_model)

    # Add synapse models testing various ways of reading PSM vars
    s_post_learn_post_sparse_pop = model.add_synapse_population(
        "PostLearnPostSparseSynapses", "SPARSE",
        post_n_pop, pre_n_pop,
        init_weight_update(post_learn_post_weight_update_model, {}, {"w": float_min},
                           psm_var_refs={"s": "s"}),
        init_postsynaptic(pattern_postsynaptic_model, {}, {"s": float_min, "shift": shift}),
        init_sparse_connectivity("OneToOne"))
    s_post_learn_post_sparse_pop.back_prop_delay_steps = delay
    s_post_sim_sparse_pop = model.add_synapse_population(
        "PostSimSparseSynapses", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(post_sim_weight_update_model, {}, {"w": float_min},
                           psm_var_refs={"s": "s"}),
        init_postsynaptic(pattern_postsynaptic_model, {}, {"s": float_min, "shift": shift}),
        init_sparse_connectivity("OneToOne"))
    s_post_sim_sparse_pop.back_prop_delay_steps = delay
    
    s_post_event_sparse_pop = model.add_synapse_population(
        "PostEventSparseSynapses", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(post_event_weight_update_model, {}, {"w": float_min},
                           psm_var_refs={"s": "s"}),
        init_postsynaptic(pattern_postsynaptic_model, {}, {"s": float_min, "shift": shift}),
        init_sparse_connectivity("OneToOne"))
    s_post_event_sparse_pop.back_prop_delay_steps = delay

    s_post_learn_post_sparse_delay_pop = model.add_synapse_population(
        "PostLearnPostSparseDelaySynapses", "SPARSE",
        post_n_pop, pre_n_pop,
        init_weight_update(post_learn_post_delay_weight_update_model, {"d": delay}, {"w": float_min},
                           psm_var_refs={"s": "s"}),
        init_postsynaptic(pattern_postsynaptic_model, {}, {"s": float_min, "shift": shift}),
        init_sparse_connectivity("OneToOne"))
    s_post_learn_post_sparse_delay_pop.max_dendritic_delay_timesteps = delay + 1
    s_post_sim_sparse_delay_pop = model.add_synapse_population(
        "PostSimSparseDelaySynapses", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(post_sim_delay_weight_update_model, {"d": delay}, {"w": float_min},
                           psm_var_refs={"s": "s"}),
        init_postsynaptic(pattern_postsynaptic_model, {}, {"s": float_min, "shift": shift}),
        init_sparse_connectivity("OneToOne"))
    s_post_sim_sparse_delay_pop.max_dendritic_delay_timesteps = delay + 1
    s_post_event_sparse_delay_pop = model.add_synapse_population(
        "PostEventSparseDelaySynapses", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(post_event_delay_weight_update_model, {"d": delay}, {"w": float_min},
                           psm_var_refs={"s": "s"}),
        init_postsynaptic(pattern_postsynaptic_model, {}, {"s": float_min, "shift": shift}),
        init_sparse_connectivity("OneToOne"))
    s_post_event_sparse_delay_pop.max_dendritic_delay_timesteps = delay + 1

    # Build model and load
    model.build()
    model.load()

    # Pull all synaptic connectivity from device
    synapse_groups = [s_post_sim_sparse_pop, 
                      s_post_learn_post_sparse_pop,
                      s_post_event_sparse_pop,
                      s_post_learn_post_sparse_delay_pop,
                      s_post_sim_sparse_delay_pop,
                      s_post_event_sparse_delay_pop]
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
