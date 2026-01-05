import pytest

from pygenn import GeNNModel, create_neuron_model


def test_model_timing_counters_nonzero():
    """
    Ensure that at least one model timing counter is non-zero
    after running a simple simulation.

    Timing must be explicitly enabled in GeNN for counters
    to be populated.
    """
    model = GeNNModel("float", "timing_test")

    # 🔑 REQUIRED: enable timing
    model.timing_enabled = True

    neuron_model = create_neuron_model(
        "neuron",
        vars=[("V", "float")],
        sim_code="V += 1.0;"
    )

    model.add_neuron_population(
        "pop",
        1,
        neuron_model,
        {},
        {"V": 0.0}
    )

    model.build()
    model.load()

    for _ in range(5):
        model.step_time()

    timings = [
        model.neuron_update_time,
        model.presynaptic_update_time,
        model.postsynaptic_update_time,
        model.synapse_dynamics_time,
        model.init_time,
        model.init_sparse_time,
    ]

    assert any(t > 0.0 for t in timings), (
        "Expected at least one model timing counter to be non-zero "
        "after running the simulation with timing enabled"
    )
