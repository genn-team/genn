import pytest
from pygenn import (
    types,
    init_weight_update,
    init_postsynaptic,
    init_sparse_connectivity,
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

    # Build and simulate
    model.build()
    model.load()

    for _ in range(100):
        model.step_time()

    # Timing counters (must be accessible and numeric)
    timing_counters = [
        model.neuron_update_time,
        model.init_time,
        model.init_sparse_time,
        model.presynaptic_update_time,
        model.postsynaptic_update_time,
    ]

    for value in timing_counters:
        assert isinstance(value, (int, float)), (
        "Expected timing counter to be numeric"
    )
        assert value >= 0.0, (
        "Expected timing counter to be non-negative"
    )
