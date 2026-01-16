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
)


def test_model_load_unload_cycle(make_model, backend):
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
    model = make_model("float", "test_lifecycle", backend)

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

    # Check neuron population variables
    for var_name, var in pre.vars.items():
        assert var._view is None, f"pre.vars[{var_name}]._view not cleared"
        assert var._array is None, f"pre.vars[{var_name}]._array not cleared"

    for var_name, var in post.vars.items():
        assert var._view is None, f"post.vars[{var_name}]._view not cleared"
        assert var._array is None, f"post.vars[{var_name}]._array not cleared"

    # Check synapse population variables
    for var_name, var in syn.vars.items():
        assert var._view is None, f"syn.vars[{var_name}]._view not cleared"
        assert var._array is None, f"syn.vars[{var_name}]._array not cleared"

    # Check current source variables
    for var_name, var in cs.vars.items():
        assert var._view is None, f"cs.vars[{var_name}]._view not cleared"
        assert var._array is None, f"cs.vars[{var_name}]._array not cleared"

    # Check custom update variables
    cu_data = model.custom_updates["TestUpdate"]
    for var_name, var in cu_data.vars.items():
        assert var._view is None, f"custom_update.vars[{var_name}]._view not cleared"
        assert var._array is None, f"custom_update.vars[{var_name}]._array not cleared"

    # ========== RELOAD ==========
    model.load()
    assert model._loaded == True

    # Verify variables are reloaded (have views/arrays again)
    assert pre.vars["V"]._view is not None
    assert pre.vars["V"]._array is not None

    # Reset state to initial conditions for reproducibility
    pre.vars["V"].view[:] = -70.0
    pre.vars["RefracTime"].view[:] = 0.0
    post.vars["V"].view[:] = -70.0
    post.vars["RefracTime"].view[:] = 0.0
    pre.vars["V"].push_to_device()
    pre.vars["RefracTime"].push_to_device()
    post.vars["V"].push_to_device()
    post.vars["RefracTime"].push_to_device()

    # Run simulation again
    for _ in range(10):
        model.step_time()
    model.custom_update("CustomUpdate")

    # Get reloaded values
    reloaded_pre_V = pre.vars["V"].view[:].copy()
    reloaded_post_V = post.vars["V"].view[:].copy()

    # Verify reloaded model produces valid results
    assert np.isfinite(reloaded_pre_V).all(), "Reloaded pre values contain NaN/Inf"
    assert np.isfinite(reloaded_post_V).all(), "Reloaded post values contain NaN/Inf"
