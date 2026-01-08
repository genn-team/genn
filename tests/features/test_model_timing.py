import pytest
from pygenn import (
    types,
    init_weight_update,
    init_postsynaptic,
    init_sparse_connectivity,
    create_weight_update_model,
    create_custom_update_model,
    create_var_ref,
)


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_model_timing_counters_accessible(make_model, backend, precision):
    """
    Verify that model timing counters are accessible when timing is enabled.

    This test builds a minimal model using built-in neuron (LIF) and synapse (STDP)
    models, runs a short simulation, and checks that all documented timing counters
    are present and numeric. Assertions are backend-agnostic.
    """
    # Create model
    model = make_model(
        precision,
        "test_model_timing_counters_accessible",
        backend=backend,
    )

    # Enable timing explicitly
    model.timing = True

    # Built-in LIF neuron model parameters (emits spikes)
    lif_params = {
        "C": 1.0,
        "TauM": 20.0,
        "Vrest": -40.0,
        "Vreset": -60.0,
        "Vthresh": -50.0,
        "Ioffset": 0.0,
        "TauRefrac": 5.0,
    }

    lif_init = {
        "V": -60.0,
        "RefracTime": 0.0,
    }

    model.add_neuron_population(
        "pop_pre",
        8,
        "LIF",
        lif_params,
        lif_init,
    )

    model.add_neuron_population(
        "pop_post",
        8,
        "LIF",
        lif_params,
        lif_init,
    )

    # Built-in STDP synapse model
    stdp_params = {
        "tauPlus": 20.0,
        "tauMinus": 20.0,
        "Aplus": 0.0,
        "Aminus": 0.0,
        "Wmin": 0.0,
        "Wmax": 1.0,
    }

    stdp_init = {
        "g": 1.0,
    }

    model.add_synapse_population(
        "syn",
        "SPARSE",
        model.neuron_populations["pop_pre"],
        model.neuron_populations["pop_post"],
        init_weight_update("STDP", stdp_params, stdp_init),
        init_postsynaptic("DeltaCurr", {}),
        init_sparse_connectivity("FixedProbability", {"prob": 1.0}),
    )

    # Add synapse population with dynamics to test synapse_dynamics_time
    stdp_dyn_model = create_weight_update_model(
        "stdp_with_dynamics",
        vars=[("g", "scalar")],
        pre_spike_syn_code="",  # Weight update happens in synapse_dynamics
        synapse_dynamics_code="g *= 0.99;"  # Simple decay
    )

    model.add_synapse_population(
        "syn_dynamics",
        "DENSE",
        model.neuron_populations["pop_pre"],
        model.neuron_populations["pop_post"],
        init_weight_update(stdp_dyn_model, {}, {"g": 1.0}),
        init_postsynaptic("DeltaCurr", {}),
    )

    # Add custom update to test custom update timing
    cu_model = create_custom_update_model(
        "test_update",
        update_code="V *= 0.99;",
        var_refs=[("V", "scalar")]
    )

    model.add_custom_update(
        "TestUpdate",
        "CustomUpdate",
        cu_model,
        {},
        {},  # No vars for the custom update itself
        var_refs={"V": create_var_ref(model.neuron_populations["pop_pre"], "V")},
    )

    # Build and simulate
    model.build()
    model.load()

    # Define lambda functions to access timing counters
    get_timing_counters = [
        lambda: model.neuron_update_time,
        lambda: model.init_time,
        lambda: model.init_sparse_time,
        lambda: model.presynaptic_update_time,
        lambda: model.postsynaptic_update_time,
        lambda: model.synapse_dynamics_time,
        lambda: model.get_custom_update_time("CustomUpdate"),
    ]

    # Capture initial timing values for accumulation test
    initial_times = [get_time() for get_time in get_timing_counters]

    for _ in range(100):
        model.step_time()

    # Run custom update to generate timing data
    model.custom_update("CustomUpdate")

    # Get final timing values
    final_times = [get_time() for get_time in get_timing_counters]

    # Check all timing counters
    for initial, final in zip(initial_times, final_times):
        # Must be non-negative
        assert final >= 0.0, (
            "Expected timing counter to be non-negative"
        )
        # Must accumulate over simulation
        assert final >= initial, (
            "Timing should accumulate over simulation"
        )
