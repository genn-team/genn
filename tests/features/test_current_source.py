import numpy as np
import pytest
from pygenn import types

from pygenn import (create_current_source_model,
                    create_neuron_model,
                    create_var_ref)
from pygenn import VarAccessMode

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_current_source_neuron_var_refs(make_model, backend, precision):
    # Neuron model which accumulates its input current
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        x += Isyn;
        """,
        vars=[("a", "scalar"), ("x", "scalar")])

    # Current source which injects the value of a neuron variable,
    # accessed through a neuron variable reference
    current_source_model = create_current_source_model(
        "current_source",
        injection_code=
        """
        injectCurrent(a);
        """,
        neuron_var_refs=[("a", "scalar", VarAccessMode.READ_ONLY)])

    model = make_model(precision, "test_current_source_neuron_var_refs",
                       backend=backend)
    model.dt = 1.0

    a = np.arange(10, dtype=float)
    pop = model.add_neuron_population("Pop", 10, neuron_model,
                                      {}, {"a": a, "x": 0.0})
    model.add_current_source("CurrentSource", current_source_model, pop,
                             {}, {}, {"a": create_var_ref(pop, "a")})

    model.build()
    model.load()

    num_steps = 5
    while model.timestep < num_steps:
        model.step_time()

    pop.vars["x"].pull_from_device()
    assert np.allclose(pop.vars["x"].view, num_steps * a)
