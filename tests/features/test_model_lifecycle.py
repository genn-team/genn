"""Test model lifecycle (load/unload/reload)

Tests for Issue: Verify that models can be cleanly unloaded and reloaded,
with proper cleanup of variable references to the shared library.
"""
import pytest
import numpy as np
from pygenn import (
    init_weight_update,
    init_postsynaptic,
    init_sparse_connectivity,
    create_custom_update_model,
    create_var_ref,
    types,
)


def _check_vars_cleared(vars_dict, component_name, expected_none=True):
    """Helper to verify all variables in a dict are (or aren't) cleared.

    Args:
        vars_dict: Dictionary of variables to check
        component_name: Name of component for error messages
        expected_none: If True, assert vars are None; if False, assert not None
    """
    for var_name, var in vars_dict.items():
        if expected_none:
            assert var._view is None, f"{component_name}.vars[{var_name}]._view not cleared"
            assert var._array is None, f"{component_name}.vars[{var_name}]._array not cleared"
        else:
            assert var._view is not None, f"{component_name}.vars[{var_name}]._view is None"
            assert var._array is not None, f"{component_name}.vars[{var_name}]._array is None"


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_model_load_unload_cycle(make_model, backend, precision):
    """Test that models can be unloaded and reloaded correctly.

    This test creates a comprehensive model with:
    - Neuron populations (pre and post)
    - Synapse populations with sparse connectivity (STDP)
    - Current sources
    - Custom updates

    Then verifies:
    1. Model loads and runs correctly
    2. After unload, all variable views/arrays are None
    3. Model can be reloaded
    4. Reloaded model runs correctly
    """
    # Create comprehensive model
    model = make_model(precision, "test_lifecycle", backend)

    # LIF neuron parameters
    lif_params = {
        "C": 1.0,
        "TauM": 20.0,
        "Vrest": -70.0,
        "Vreset": -70.0,
        "Vthresh": -50.0,
        "Ioffset": 0.0,
        "TauRefrac": 5.0,
    }
    lif_init = {"V": -70.0, "RefracTime": 0.0}

    # Add neuron populations
    pre = model.add_neuron_population("pre", 10, "LIF", lif_params, lif_init)
    post = model.add_neuron_population("post", 10, "LIF", lif_params, lif_init)

    # Add synapses with STDP
    stdp_params = {
        "tauPlus": 20.0,
        "tauMinus": 20.0,
        "Aplus": 0.01,
        "Aminus": 0.01,
        "Wmin": 0.0,
        "Wmax": 1.0,
    }
    stdp_init = {"g": 0.5}

    syn = model.add_synapse_population(
        "synapses",
        "SPARSE",
        pre,
        post,
        init_weight_update("STDP", stdp_params, stdp_init),
        init_postsynaptic("DeltaCurr", {}),
        init_sparse_connectivity("FixedProbability", {"prob": 0.5}),
    )

    # Add current source
    cs = model.add_current_source("current", "DC", pre, {"amp": 5.0}, {})

    # Add custom update
    cu_model = create_custom_update_model(
        "test_update", update_code="V *= 0.99;", var_refs=[("V", "scalar")]
    )
    model.add_custom_update(
        "TestUpdate",
        "CustomUpdate",
        cu_model,
        {},
        {},
        var_refs={"V": create_var_ref(pre, "V")},
    )

    # Build model
    model.build()

    # ========== FIRST LOAD ==========
    model.load()
    assert model._loaded == True

    # Verify variables are loaded (have views/arrays)
    assert pre.vars["V"]._view is not None
    assert pre.vars["V"]._array is not None

    # Run simulation
    for _ in range(10):
        model.step_time()
    model.custom_update("CustomUpdate")

    # ========== UNLOAD ==========
    model.unload()

    # Verify model is unloaded
    assert model._loaded == False
    assert model._runtime is None

    # CRITICAL: Verify all variable references are cleared
    # (This is what prevents the shared library from unloading properly)
    _check_vars_cleared(pre.vars, "pre")
    _check_vars_cleared(post.vars, "post")
    _check_vars_cleared(syn.vars, "synapses")
    _check_vars_cleared(cs.vars, "current_source")
    _check_vars_cleared(model.custom_updates["TestUpdate"].vars, "custom_update")

    # ========== RELOAD ==========
    model.load()
    assert model._loaded == True

    # Verify variables are reloaded (have views/arrays again)
    _check_vars_cleared(pre.vars, "pre", expected_none=False)
    _check_vars_cleared(post.vars, "post", expected_none=False)

    # Run simulation again to verify reloaded model works
    for _ in range(10):
        model.step_time()
    model.custom_update("CustomUpdate")


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_access_before_load(make_model, backend, precision):
    """Test that accessing simulation state before the model is
    loaded (or after it is unloaded) gives explicit errors."""
    model = make_model(precision, "test_access_before_load", backend)
    model.dt = 1.0

    pop = model.add_neuron_population("pop", 10, "SpikeSourceArray", {},
                                      {"startSpike": 0, "endSpike": 0})
    pop.extra_global_params["spikeTimes"].set_init_values(np.zeros(1))

    def check_not_loaded():
        # Simulation time
        with pytest.raises(Exception, match="has to be loaded"):
            model.t
        with pytest.raises(Exception, match="has to be loaded"):
            model.timestep
        with pytest.raises(Exception, match="has to be loaded"):
            model.timestep = 0

        # Variables and extra global parameters
        for array in [pop.vars["startSpike"],
                      pop.extra_global_params["spikeTimes"]]:
            with pytest.raises(RuntimeError, match="has to be loaded"):
                array.values
            with pytest.raises(RuntimeError, match="has to be loaded"):
                array.values = 0
            with pytest.raises(RuntimeError, match="has to be loaded"):
                array.push_to_device()
            with pytest.raises(RuntimeError, match="has to be loaded"):
                array.pull_from_device()

    # Check errors are given before model is built and before it is loaded
    check_not_loaded()
    model.build()
    check_not_loaded()

    # Load model and check everything can now be accessed
    model.load()
    model.step_time()
    assert model.t == 1.0
    assert model.timestep == 1
    model.timestep = 0
    assert model.timestep == 0
    pop.vars["startSpike"].values = 1
    pop.vars["startSpike"].push_to_device()
    pop.vars["startSpike"].pull_from_device()
    assert np.all(pop.vars["startSpike"].values == 1)
    assert np.all(pop.extra_global_params["spikeTimes"].values == 0.0)

    # Unload model and check errors are given again
    model.unload()
    check_not_loaded()

    # Reload so model can be unloaded when test finishes
    model.load()
