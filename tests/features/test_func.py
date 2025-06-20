import numpy as np
import pytest
from pygenn import types

from pygenn import create_neuron_model, init_var

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_func(make_model, backend, precision, batch_size):
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        s = signbit(x);
        """,
        vars=[("s", "scalar"), ("x", "scalar")])

    model = make_model(precision, "test_func", backend=backend)
    model.batch_size = batch_size

    # Add neuron populations
    var_init = {"s": 0.0, "x": init_var("Normal", {"mean": 0.0, "sd": 1.0})}
    n_pop = model.add_neuron_population("Neurons", 1000, neuron_model, 
                                        {}, var_init)

    # Build model and load
    model.build()
    model.load()
    
    # Simulate single step
    model.step_time()
    
    # Pull variables from device
    n_pop.vars["s"].pull_from_device()
    n_pop.vars["x"].pull_from_device()
    
    # Check signbit function performs correct function
    assert np.all((n_pop.vars["x"].values < 0.0) == n_pop.vars["s"].values)