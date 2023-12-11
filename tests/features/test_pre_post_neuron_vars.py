import numpy as np
import pytest
from pygenn import types
from scipy import stats

from pygenn.genn import VarAccess, VarAccessMode
from pygenn import (create_neuron_model, create_var_ref,
                    create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_weight_update, init_var)

# Weight update models which copy a PRESYNAPTIC neuron variables
# into synapse during various kinds of synaptic event
pre_learn_post_weight_update_model = create_weight_update_model(
    "pre_learn_post_weight_update",
    var_name_types=[("w", "scalar")],
    pre_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],
    
    learn_post_code=
    """
    w = s;
    """)

pre_sim_weight_update_model = create_weight_update_model(
    "pre_sim_weight_update",
    var_name_types=[("w", "scalar")],
    pre_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    sim_code=
    """
    w = s;
    """)

pre_event_weight_update_model = create_weight_update_model(
    "pre_sim_weight_update",
    var_name_types=[("w", "scalar")],
    pre_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    event_code=
    """
    w = s;
    """,
    event_threshold_condition_code=
    """
    t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
    """)

# Weight update models which copy a POSTSYNAPTIC neuron variables
# into synapse during various kinds of synaptic event
post_sim_weight_update_model = create_weight_update_model(
    "post_sim_weight_update",
    var_name_types=[("w", "scalar")],
    post_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],
    
    sim_code=
    """
    w = s;
    """)

post_learn_post_weight_update_model = create_weight_update_model(
    "post_learn_post_weight_update",
    var_name_types=[("w", "scalar")],
    post_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],
    
    learn_post_code=
    """
    w = s;
    """)

post_event_weight_update_model = create_weight_update_model(
    "pre_sim_weight_update",
    var_name_types=[("w", "scalar")],
    post_neuron_var_refs=[("s", "scalar", VarAccessMode.READ_ONLY)],

    event_code=
    """
    w = s;
    """,
    event_threshold_condition_code=
    """
    t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
    """)
    
@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
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
        var_name_types=[("s", "scalar"), ("shift", "scalar", VarAccess.READ_ONLY)])

    # Neuron model which updates a state variable with time + per-neuron shift
    shift_pattern_neuron_model = create_neuron_model(
        "shift_pattern_neuron",
        sim_code=
        """
        s = t + shift;
        """,
        var_name_types=[("s", "scalar"), ("shift", "scalar", VarAccess.READ_ONLY)])


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
        "PreLearnPostSparseSynapses", "SPARSE", delay,
        post_n_pop, pre_n_pop,
        init_weight_update(pre_learn_post_weight_update_model, {}, {"w": float_min},
                           pre_var_refs={"s": create_var_ref(post_n_pop, "s")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    s_pre_sim_sparse_pop = model.add_synapse_population(
        "PreSimSparseSynapses", "SPARSE", delay,
        pre_n_pop, post_n_pop,
        init_weight_update(pre_sim_weight_update_model, {}, {"w": float_min},
                           pre_var_refs={"s": create_var_ref(pre_n_pop, "s")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    s_pre_event_sparse_pop = model.add_synapse_population(
        "PreEventSparseSynapses", "SPARSE", delay,
        pre_n_pop, post_n_pop,
        init_weight_update(pre_event_weight_update_model, {}, {"w": float_min},
                           pre_var_refs={"s": create_var_ref(pre_n_pop, "s")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
        
    # Add synapse models testing various ways of reading post WU vars
    s_post_learn_post_sparse_pop = model.add_synapse_population(
        "PostLearnPostSparseSynapses", "SPARSE", 0,
        post_n_pop, pre_n_pop,
        init_weight_update(post_learn_post_weight_update_model, {}, {"w": float_min},
                           post_var_refs={"s": create_var_ref(pre_n_pop, "s")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    s_post_learn_post_sparse_pop.back_prop_delay_steps = delay
    s_post_sim_sparse_pop = model.add_synapse_population(
        "PostSimSparseSynapses", "SPARSE", 0,
        pre_n_pop, post_n_pop,
        init_weight_update(post_sim_weight_update_model, {}, {"w": float_min},
                           post_var_refs={"s": create_var_ref(post_n_pop, "s")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    s_post_sim_sparse_pop.back_prop_delay_steps = delay
    
    s_post_event_sparse_pop = model.add_synapse_population(
        "PostEventSparseSynapses", "SPARSE", 0,
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

if __name__ == '__main__':
    test_pre_post_neuron_var("cuda", types.Float, 20)
