import pytest

from pygenn import types
from pygenn.genn_model import (
    create_neuron_model,
    init_weight_update,
    init_postsynaptic,
    init_sparse_connectivity,
)


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_model_timing_counters_accessible(make_model, backend, precision):
    """
    Verify that model timing counters are available and accessible when
    timing is enabled.

    Timing counters are backend-dependent and may legitimately remain zero
    on some backends (e.g. single-threaded CPU). Therefore, this test
    verifies that counters exist, are numeric, and can be queried without
    error, rather than asserting non-zero values.
    """

    # Create model using standard test fixture
    model = make_model(
        precision,
        "test_model_timing_counters_accessible",
        backend=backend,
    )

    # Enable timing explicitly
    model.timing = True

    # Neuron model
    neuron_model = create_neuron_model(
        "neuron",
        vars=[("V", precision)],
        sim_code="V += 1.0;"
    )

    model.add_neuron_population(
        "pop_pre",
        4,
        neuron_model,
        {},
        {"V": 0.0}
    )

    model.add_neuron_population(
        "pop_post",
        4,
        neuron_model,
        {},
        {"V": 0.0}
    )

    # Synapse population
    model.add_synapse_population(
        "syn",
        "SPARSE",
        model.neuron_populations["pop_pre"],
        model.neuron_populations["pop_post"],
        init_weight_update(
            "StaticPulseConstantWeight",
            {"g": 1.0}
        ),
        init_postsynaptic(
            "DeltaCurr",
            {}
        ),
        init_sparse_connectivity(
            "FixedProbability",
            {"prob": 1.0}
        )
    )

    # Build and run model
    model.build()
    model.load()

    for _ in range(5):
        model.step_time()


    # Assertions: counters exist and are numeric
    timing_counters = [
        model.neuron_update_time,
        model.init_time,
        model.init_sparse_time,
        model.presynaptic_update_time,
        model.postsynaptic_update_time,
        model.synapse_dynamics_time,
    ]

    for value in timing_counters:
        assert isinstance(value, (int, float)), (
            "Expected timing counter to be numeric"
        )
