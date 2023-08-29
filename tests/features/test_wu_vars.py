import numpy as np
import pytest
from pygenn import types
from scipy import stats

from pygenn import GeNNModel

from pygenn.genn import NeuronVarAccess
from pygenn import (create_neuron_model,
                    create_weight_update_model,
                    init_sparse_connectivity, init_var)

# Neuron model which fires every timestep
# **NOTE** this is just so sim_code fires every timestep
always_spike_neuron_model = create_neuron_model(
    "always_spike_neuron",
    threshold_condition_code="true")

# Neuron model which fires at t = id ms and every 10 ms after that
pattern_spike_neuron_model = create_neuron_model(
    "pattern_spike_neuron",
    threshold_condition_code=
    """
    t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
    """)

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
@pytest.mark.parametrize("fuse", [True, False])
@pytest.mark.parametrize("delay", [0, 20])
def test_wu_var(backend, precision, fuse, delay):
    # Weight update models which update a PRESYNAPTIC variable 
    # when a PRESYNAPTIC spike is received and copy it into synapse 
    # during various kinds of synaptic event
    pre_learn_post_weight_update_model = create_weight_update_model(
        "pre_learn_post_weight_update",
        var_name_types=[("w", "scalar")],
        pre_var_name_types=[("s", "scalar")],
        
        learn_post_code=
        """
        w = s;
        """,
        pre_spike_code=
        """
        s = t;
        """)
    
    pre_sim_weight_update_model = create_weight_update_model(
        "pre_sim_weight_update",
        var_name_types=[("w", "scalar")],
        pre_var_name_types=[("s", "scalar")],
        
        sim_code=
        """
        w = s;
        """,
        pre_spike_code=
        """
        s = t;
        """)
    
    pre_synapse_dynamics_weight_update_model = create_weight_update_model(
        "pre_synapse_dynamics_weight_update",
        var_name_types=[("w", "scalar")],
        pre_var_name_types=[("s", "scalar")],
        
        synapse_dynamics_code=
        """
        w = s;
        """,
        pre_spike_code=
        """
        s = t;
        """)
    
    # Weight update models which update a POSTSYNAPTIC variable 
    # when a POSTSYNAPTIC spike is received and copy it into synapse 
    # during various kinds of synaptic event
    post_learn_post_weight_update_model = create_weight_update_model(
        "post_learn_post_weight_update",
        var_name_types=[("w", "scalar")],
        post_var_name_types=[("s", "scalar")],
        
        learn_post_code=
        """
        w = s;
        """,
        post_spike_code=
        """
        s = t;
        """)
    
    post_sim_weight_update_model = create_weight_update_model(
        "post_sim_weight_update",
        var_name_types=[("w", "scalar")],
        post_var_name_types=[("s", "scalar")],
        
        sim_code=
        """
        w = s;
        """,
        post_spike_code=
        """
        s = t;
        """)
    
    post_synapse_dynamics_weight_update_model = create_weight_update_model(
        "post_synapse_dynamics_weight_update",
        var_name_types=[("w", "scalar")],
        post_var_name_types=[("s", "scalar")],
        
        synapse_dynamics_code=
        """
        w = s;
        """,
        post_spike_code=
        """
        s = t;
        """)

    model = GeNNModel(precision, "test_wu_var", backend=backend)
    model.dt = 1.0
    model.fuse_pre_post_weight_update_models = fuse
    model.fuse_postsynaptic_models = fuse

    # Create pre and postsynaptic neuron populations
    pre_n_pop = model.add_neuron_population("PreNeurons", 10, always_spike_neuron_model, 
                                            {}, {})
  
    post_n_pop = model.add_neuron_population("PostNeurons", 10, pattern_spike_neuron_model, 
                                             {}, {})

    # Add synapse models testing various ways of reading presynaptic WU vars
    float_min = np.finfo(np.float32).min
    s_pre_learn_post_sparse_pop = model.add_synapse_population(
        "PreLearnPostSparseSynapses", "SPARSE", delay,
        post_n_pop, pre_n_pop,
        pre_learn_post_weight_update_model, {}, {"w": float_min}, {"s": float_min}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))

    s_pre_sim_sparse_pop = model.add_synapse_population(
        "PreSimSparseSynapses", "SPARSE", delay,
        post_n_pop, pre_n_pop,
        pre_sim_weight_update_model, {}, {"w": float_min}, {"s": float_min}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))

    s_pre_synapse_dynamics_sparse_pop = model.add_synapse_population(
        "PreSynapseDynamicsSparseSynapses", "SPARSE", delay,
        post_n_pop, pre_n_pop,
        pre_synapse_dynamics_weight_update_model, {}, {"w": float_min}, {"s": float_min}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
    
    # Add synapse models testing various ways of reading post WU vars
    s_post_learn_post_sparse_pop = model.add_synapse_population(
        "PostLearnPostSparseSynapses", "SPARSE", 0,
        pre_n_pop, post_n_pop,
        post_learn_post_weight_update_model, {}, {"w": float_min}, {}, {"s": float_min},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
    s_post_learn_post_sparse_pop.back_prop_delay_steps = delay
    
    s_post_sim_sparse_pop = model.add_synapse_population(
        "PostSimSparseSynapses", "SPARSE", 0,
        pre_n_pop, post_n_pop,
        post_sim_weight_update_model, {}, {"w": float_min}, {}, {"s": float_min},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
    s_post_sim_sparse_pop.back_prop_delay_steps = delay
    
    s_post_synapse_dynamics_sparse_pop = model.add_synapse_population(
        "PostSynapseDynamicsSparseSynapses", "SPARSE", 0,
        pre_n_pop, post_n_pop,
        post_synapse_dynamics_weight_update_model, {}, {"w": float_min}, {}, {"s": float_min},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
    s_post_synapse_dynamics_sparse_pop.back_prop_delay_steps = delay

    # Build model and load
    model.build()
    model.load()

    # Pull all synaptic connectivity from device
    synapse_groups = [s_post_sim_sparse_pop, s_post_learn_post_sparse_pop,
                      s_post_synapse_dynamics_sparse_pop, 
                      s_pre_sim_sparse_pop, s_pre_learn_post_sparse_pop,
                      s_pre_synapse_dynamics_sparse_pop]
    for s in synapse_groups:
        s.pull_connectivity_from_device()
    
    while model.timestep < 100:
        model.step_time()
        
        # Calculate time of spikes we SHOULD be reading
        # **NOTE** we delay by delay + 2 timesteps because:
        # 1) delay = delay
        # 2) spike times are read in synapse kernel one timestep AFTER being emitted
        # 3) t is incremented one timestep at te end of StepGeNN
        delayed_time = np.arange(10) + (10.0 * np.floor((model.t - delay - 2.0 - np.arange(10)) / 10.0))
        delayed_time[delayed_time < 0.0] = float_min
        
        # Loop through synapse groups and compare value of w with delayed time
        for s in synapse_groups:
            s.pull_var_from_device("w")
            w_value = s.get_var_values("w")
            if not np.allclose(delayed_time, w_value):
                assert False, f"{s.name} var has wrong value ({w_value} rather than {delayed_time})"

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
@pytest.mark.parametrize("fuse", [True, False])
@pytest.mark.parametrize("delay", [0, 20])
def test_wu_var_cont(backend, precision, fuse, delay):
    # Weight update models which update a PRESYNAPTIC variable 
    # when a PRESYNAPTIC spike is received and copy it into synapse 
    # during various kinds of synaptic event
    pre_learn_post_weight_update_model = create_weight_update_model(
        "pre_learn_post_weight_update",
        var_name_types=[("w", "scalar")],
        pre_var_name_types=[("s", "scalar"), ("shift", "scalar", NeuronVarAccess.READ_ONLY)],
        
        learn_post_code=
        """
        w = s;
        """,
        pre_dynamics_code=
        """
        s = t + shift;
        """)

    pre_sim_weight_update_model = create_weight_update_model(
        "pre_sim_weight_update",
        var_name_types=[("w", "scalar")],
        pre_var_name_types=[("s", "scalar"), ("shift", "scalar", NeuronVarAccess.READ_ONLY)],
        
        sim_code=
        """
        w = s;
        """,
        pre_dynamics_code=
        """
        s = t + shift;
        """)

    # Weight update models which update a POSTSYNAPTIC variable 
    # when a POSTSYNAPTIC spike is received and copy it into synapse 
    # during various kinds of synaptic event
    post_learn_post_weight_update_model = create_weight_update_model(
        "post_learn_post_weight_update",
        var_name_types=[("w", "scalar")],
        post_var_name_types=[("s", "scalar"), ("shift", "scalar", NeuronVarAccess.READ_ONLY)],
        
        learn_post_code=
        """
        w = s;
        """,
        post_dynamics_code=
        """
        s = t + shift;
        """)
    
    post_sim_weight_update_model = create_weight_update_model(
        "post_sim_weight_update",
        var_name_types=[("w", "scalar")],
        post_var_name_types=[("s", "scalar"), ("shift", "scalar", NeuronVarAccess.READ_ONLY)],
        
        sim_code=
        """
        w = s;
        """,
        post_dynamics_code=
        """
        s = t + shift;
        """)

    model = GeNNModel(precision, "test_wu_var_cont", backend=backend)
    model.dt = 1.0
    model.fuse_pre_post_weight_update_models = fuse
    model.fuse_postsynaptic_models = fuse

    # Create pre and postsynaptic neuron populations
    pre_n_pop = model.add_neuron_population("PreNeurons", 10, pattern_spike_neuron_model, 
                                            {}, {})
  
    post_n_pop = model.add_neuron_population("PostNeurons", 10, "SpikeSource", 
                                             {}, {})

    # Add synapse models testing various ways of reading presynaptic WU vars
    float_min = np.finfo(np.float32).min
    shift = np.arange(0.0, 100.0, 10.0)
    s_pre_learn_post_sparse_pop = model.add_synapse_population(
        "PreLearnPostSparseSynapses", "SPARSE", delay,
        post_n_pop, pre_n_pop,
        pre_learn_post_weight_update_model, {}, {"w": float_min}, {"s": float_min, "shift": shift}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
    s_pre_sim_sparse_pop = model.add_synapse_population(
        "PreSimSparseSynapses", "SPARSE", delay,
        pre_n_pop, post_n_pop, 
        pre_sim_weight_update_model, {}, {"w": float_min}, {"s": float_min, "shift": shift}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
        
    # Add synapse models testing various ways of reading post WU vars
    s_post_learn_post_sparse_pop = model.add_synapse_population(
        "PostLearnPostSparseSynapses", "SPARSE", 0,
        post_n_pop, pre_n_pop,
        post_learn_post_weight_update_model, {}, {"w": float_min}, {}, {"s": float_min, "shift": shift},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
    s_post_learn_post_sparse_pop.back_prop_delay_steps = delay
    s_post_sim_sparse_pop = model.add_synapse_population(
        "PostSimSparseSynapses", "SPARSE", 0,
        pre_n_pop, post_n_pop,
        post_sim_weight_update_model, {}, {"w": float_min}, {}, {"s": float_min, "shift": shift},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
    s_post_sim_sparse_pop.back_prop_delay_steps = delay

    # Build model and load
    model.build()
    model.load()

    # Pull all synaptic connectivity from device
    synapse_groups = [s_post_sim_sparse_pop, 
                      s_post_learn_post_sparse_pop,
                      s_pre_sim_sparse_pop, 
                      s_pre_learn_post_sparse_pop]
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
            s.pull_var_from_device("w")
            w_value = s.get_var_values("w")
            if not np.allclose(delayed_time, w_value):
                assert False, f"{s.name} var has wrong value ({w_value} rather than {delayed_time})"

if __name__ == '__main__':
    test_wu_var_cont("cuda", types.Float, True, 0)
