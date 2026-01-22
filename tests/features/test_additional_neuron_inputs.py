import numpy as np
import pytest
from pygenn import types

from pygenn import (
    create_neuron_model,
    create_current_source_model
)

@pytest.mark.parametrize("precision", [types.Float, types.Double])
def test_additional_neuron_inputs(make_model, backend, precision):
    # Neuron with TWO inputs: Isyn (default) + Iext (additional)
    neuron_model = create_neuron_model(
        "multi_input_neuron",
        vars=[("v", "scalar")],
        sim_code=
        """
        v += Isyn + Iext;
        """
    )

    # Current source writing to the additional input
    current_source_model = create_current_source_model(
        "ext_current",
        vars=[("Iext", "scalar")],
        injection_code=
        """
        Iext = 1.0;
        """
    )

    model = make_model(precision, "test_additional_neuron_inputs", backend=backend)
    model.dt = 1.0

    # One neuron, initial v = 0
    pop = model.add_neuron_population(
        "N", 1, neuron_model, {}, {"v": 0.0}
    )

    # Inject external current
    model.add_current_source(
        "Ext", current_source_model, pop
    )

    # Build and load
    model.build()
    model.load()

    # Step once
    model.step_time()

    # Pull value from device
    pop.vars["v"].pull_from_device()

    # v should increase due to the additional input
    assert np.allclose(pop.vars["v"].values, 1.0)
