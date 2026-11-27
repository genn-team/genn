import numpy as np
import pytest
from pygenn import types

from pygenn import (create_current_source_model,
                    create_neuron_model,
                    create_var_ref)
from pygenn import VarAccessMode

# Initial value of additional input variable
ISYN_A_INIT = 3.0

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_current_source(make_model, backend, precision):
    # Neuron model which accumulates its standard
    # and additional input variables
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        x += Isyn;
        y += IsynA;
        """,
        vars=[("a", "scalar"), ("x", "scalar"), ("y", "scalar")],
        additional_input_vars=[("IsynA", "scalar", ISYN_A_INIT)])

    # Current source which injects the value of a neuron variable,
    # accessed through a neuron variable reference
    current_source_model = create_current_source_model(
        "current_source",
        injection_code=
        """
        injectCurrent(a);
        """,
        neuron_var_refs=[("a", "scalar", VarAccessMode.READ_ONLY)])

    model = make_model(precision, "test_current_source",
                       backend=backend)
    model.dt = 1.0

    a = np.arange(10, dtype=float)
    pop = model.add_neuron_population("Pop", 10, neuron_model,
                                      {}, {"a": a, "x": 0.0, "y": 0.0})
    model.add_current_source("CurrentSource", current_source_model, pop,
                             {}, {}, {"a": create_var_ref(pop, "a")})

    # Add second current source targetting additional input variable
    cs_a = model.add_current_source("CurrentSourceA", current_source_model, pop,
                                    {}, {}, {"a": create_var_ref(pop, "a")})
    cs_a.target_var = "IsynA"

    model.build()
    model.load()

    num_steps = 5
    while model.timestep < num_steps:
        model.step_time()

    # Additional input variable should be reset to its initial value every timestep
    pop.vars["x"].pull_from_device()
    pop.vars["y"].pull_from_device()
    assert np.allclose(pop.vars["x"].view, num_steps * a)
    assert np.allclose(pop.vars["y"].view, num_steps * (ISYN_A_INIT + a))
