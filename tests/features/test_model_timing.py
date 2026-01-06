import pytest

from pygenn import (
    GeNNModel,
    types,
    create_neuron_model,
    init_weight_update,
    init_postsynaptic,
    init_sparse_connectivity,
)

# Test: model timing counters are instantiated and accessible

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_model_timing_counters_accessible(make_model, backend, precision):
    """
    Verify that model timing counters are instantiated when timing is enabled.

    This test uses built-in neuron and synapse models to exercise as many
    timing paths as possible across backends.
    """

    # Create model
    model = make_model(
        precision,
        "test_model_timing_counters_accessible",
        backend=backend,
    )

    # Explicitly enable timing
    model.timing = True

    # Neuron model: built-in LIF (emits spikes)
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

    neuron_model = create_neuron_model(
        "LIF",
        params=lif_params,
        vars=[("V", precision), ("RefracTime", precision)],
    )

    model.add_neuron_population(
        "pop_pre",
        8,
        neuron_model,
        lif_params,
        lif_init,
    )

    model.add_neuron_population(
        "pop_post",
        8,
        neuron_model,
        lif_params,
        lif_init,
    )

    # Synapse model: built-in STDP
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

    # Build and run model
    model.build()
    model.load()

    # Run long enough to trigger timing accumulation
    for _ in range(100):
        model.step_time()

    # Assertions
    timing_counters = [
        model.neuron_update_time,
        model.init_time,
        model.init_sparse_time,
        model.presynaptic_update_time,
        model.postsynaptic_update_time,
        model.synapse_dynamics_time,  # may be zero on some backends
    ]

    for value in timing_counters:
        assert isinstance(value, (int, float)), (
            "Expected timing counter to be numeric"
        )
        assert value >= 0.0
