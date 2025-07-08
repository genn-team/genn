import numpy as np
import pytest
from pygenn import types

from pygenn import VarAccess, VarAccessMode

from pygenn import (create_custom_update_model, create_neuron_model, 
                    create_var_init_snippet, create_wu_var_ref, 
                    init_postsynaptic, init_sparse_connectivity,
                    init_var, init_weight_update)

# Neuron model which does nothing
empty_neuron_model = create_neuron_model("empty")

double_custom_update_model = create_custom_update_model(
        "double_custom_update",
         update_code=
         """
         X *= 2.0;
         """,
         var_refs=[("X", "scalar")])

weight_init_snippet = create_var_init_snippet(
        "weight_init",
        var_init_code=
        """
        value = (scalar)id_pre / id_post;
        """)

def _safe_check(a, b):
    assert (np.allclose(a[np.isfinite(a)], b[np.isfinite(b)]) 
            and np.array_equal(np.isfinite(a), np.isfinite(b)))

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_host_sparsity(make_model, backend, precision, batch_size):
    model = make_model(precision, "test_host_sparsity", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size

    # Add empty pre and postsynatic populations
    pre_pop = model.add_neuron_population("Pre", 10, empty_neuron_model)
    post_pop = model.add_neuron_population("Post", 10, empty_neuron_model)

    pre_ind = np.random.randint(10, size=20)
    post_ind = np.random.randint(10, size=20)
    g = pre_ind / post_ind

    # Connect with synapse population
    sg = model.add_synapse_population(
        "Synapses", "SPARSE",
        pre_pop, post_pop,
        init_weight_update("StaticPulse", {}, {"g": g}),
        init_postsynaptic("DeltaCurr"))
    sg.set_sparse_connections(pre_ind, post_ind)

    # Add custom update to double weights
    model.add_custom_update("DoubleG", "Double", double_custom_update_model,
                            {}, {}, {"X": create_wu_var_ref(sg, "g")})
   

    # Build model and load
    model.build()
    model.load(num_recording_timesteps=100)

    baseline_g = sg.get_sparse_pre_inds() / sg.get_sparse_post_inds()
    double_baseline_g = 2.0 * baseline_g

    # Run custom update to double
    model.custom_update("Double")

    # Pull g and check it's doubled
    sg.vars["g"].pull_from_device()
    _safe_check(sg.vars["g"].values, double_baseline_g)
    
    # Restore and push
    sg.vars["g"].values = baseline_g
    sg.vars["g"].push_to_device()

    # Run custom update to double
    model.custom_update("Double")

    # Pull g and check it's doubled
    sg.vars["g"].pull_from_device()
    _safe_check(sg.vars["g"].values, double_baseline_g)


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_device_sparsity(make_model, backend, precision, batch_size):
    model = make_model(precision, "test_device_sparsity", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size

    # Add empty pre and postsynatic populations
    pre_pop = model.add_neuron_population("Pre", 10, empty_neuron_model)
    post_pop = model.add_neuron_population("Post", 10, empty_neuron_model)

    # Connect with synapse population
    sg = model.add_synapse_population(
        "Synapses", "SPARSE",
        pre_pop, post_pop,
        init_weight_update("StaticPulse", {}, {"g": init_var(weight_init_snippet)}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("FixedNumberTotalWithReplacement", {"num": 20}))

    # Add custom update to double weights
    model.add_custom_update("DoubleG", "Double", double_custom_update_model,
                            {}, {}, {"X": create_wu_var_ref(sg, "g")})
   
    # Build model and load
    model.build()
    model.load(num_recording_timesteps=100)

    sg.pull_connectivity_from_device()
    baseline_g = sg.get_sparse_pre_inds() / sg.get_sparse_post_inds()
    double_baseline_g = 2.0 * baseline_g

    # Run custom update to double
    model.custom_update("Double")

    # Pull g and check it's doubled
    sg.vars["g"].pull_from_device()
    _safe_check(sg.vars["g"].values, double_baseline_g)
    
    # Restore and push
    sg.vars["g"].values = baseline_g
    sg.vars["g"].push_to_device()

    # Run custom update to double
    model.custom_update("Double")

    # Pull g and check it's doubled
    sg.vars["g"].pull_from_device()
    _safe_check(sg.vars["g"].values, double_baseline_g)


