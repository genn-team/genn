import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel

from pygenn.genn import SpanType, VarAccess
from pygenn import (create_neuron_model,
                    create_sparse_connect_init_snippet,
                    create_var_init_snippet,
                    create_weight_update_model,
                    init_sparse_connectivity, 
                    init_toeplitz_connectivity,
                    init_var)

# Neuron model which fires every timestep
# **NOTE** this is just so sim_code fires every timestep
always_spike_neuron_model = create_neuron_model(
    "always_spike_neuron",
    threshold_condition_code="true")

# Beuron model which fires at t = id ms and every 10 ms after that
pattern_spike_neuron_model = create_neuron_model(
    "pattern_spike_neuron",
    threshold_condition_code=
    """
    t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
    """)

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_spike_times_pre(backend, precision):
    # Weight update models which update a PRESYNAPTIC variable 
    # when a PRESYNAPTIC spike is received and copy it into synapse 
    # during various kinds of synaptic event
    weight_update_model = create_weight_update_model(
        "weight_update",
        var_name_types=[("a", "scalar"), ("b", "scalar")],
        sim_code=
        """
        a = prev_st_pre;
        """,
        learn_post_code=
        """
        b = st_pre;
        """)
    
    model = GeNNModel(precision, "test_spike_times_pre", backend=backend)
    model.dt = 1.0

    # Create pre and postsynaptic neuron populations
    pre_n_pop = model.add_neuron_population("PreNeurons", 10, pattern_spike_neuron_model, 
                                            {}, {})
  
    post_n_pop = model.add_neuron_population("PostNeurons", 10, always_spike_neuron_model, 
                                             {}, {})

    # Add synapse models testing various ways of reading presynaptic WU vars
    float_min = np.finfo(np.float32).min
    s_pre_learn_post_sparse_pop = model.add_synapse_population(
        "PreLearnPostSparseSynapses", "SPARSE", 20,
        post_n_pop, pre_n_pop,
        weight_update_model, {}, {"a": float_min, "b": float_min}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
    
    # Build model and load
    model.build()
    model.load()


if __name__ == '__main__':
    test_spike_times_pre("single_threaded_cpu", types.Float)