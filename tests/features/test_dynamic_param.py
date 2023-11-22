import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel

from pygenn import (create_current_source_model, 
                    create_custom_connectivity_update_model,
                    create_custom_update_model,
                    create_egp_ref,
                    create_neuron_model,
                    create_postsynaptic_model,
                    create_var_init_snippet,
                    create_var_ref,
                    create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_weight_update, init_var)

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_dynamic_param(backend, precision):
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        x = t + shift + input;
        """,
        params=["input"],
        var_name_types=[("x", "scalar"), ("shift", "scalar")])
    
    model = GeNNModel(precision, "test_dynamic_param", backend=backend)
    model.dt = 1.0
    
    shift = np.arange(0.0, 100.0, 10.0)
    n_pop = model.add_neuron_population("Neurons", 10, neuron_model, {"input": 0.0}, 
                                        {"x": 0.0, "shift": shift});
    n_pop.set_param_dynamic("input")

    # Build model and load
    model.build()
    model.load()
    
    while model.timestep < 100:
        correct = model.t + shift + model.t ** 2
        model.step_time()

        n_pop.vars["x"].pull_from_device()
        assert np.allclose(n_pop.vars["x"].view, correct)

        # Set dynamic parameter
        n_pop.set_dynamic_param_value("input", model.t ** 2)
        

if __name__ == '__main__':
    test_dynamic_param("single_threaded_cpu", types.Float)
