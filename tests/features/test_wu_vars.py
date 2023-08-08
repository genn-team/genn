import numpy as np
import pytest
from pygenn import types
from scipy import stats

from pygenn import GeNNModel

from pygenn import (create_neuron_model,
                    create_weight_update_model,
                    init_var)

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
@pytest.mark.parametrize("fuse", [True, False])
def test_wu_vars_post(backend, precision, fuse):
    # Presynaptic neuron model which fires every timestep
    # **NOTE** this is just so sim_code fires every timestep
    pre_neuron_model = create_neuron_model(
        "pre_neuron",
        threshold_condition_code="true")
    
    # Postsynaptic neuron model which fires at 
    # t = id ms and every 10 ms after that
    post_neuron_model = create_neuron_model(
        "post_neuron",
        threshold_condition_code=
        """
        t >= (scalar)id && fmodf(t - (scalar)id, 10.0) < 1e-4
        """)
    
    learn_post_weight_update_model = create_weight_update_model(
        "learn_post_weight_update",
        var_name_types=[("w", "scalar")],
        post_var_name_types=[("s", "scalar")],
        
        learn_post_code=
        """
        w = s;
        """,
        post_spike_code=
        """
        s = t;
        """)

    model = GeNNModel(precision, "test_post_wu_var", backend=backend)
    model.dt = 1.0
    model.fuse_pre_post_weight_update_models = fuse
    
    # Create pre and postsynaptic neuron populations
    pre_n_pop = model.add_neuron_population("PreNeurons", 10, pre_neuron_model, 
                                            {}, {})
  
    post_n_pop = model.add_neuron_population("PostNeurons", 10, post_neuron_model, 
                                             {}, {})

    # Add synapse models testing various ways of reading post WU vars
    float_min = np.finfo(np.float32).min
    s_learn_post_dense_pop = model.add_synapse_population(
        "LearnPostDenseSynapses", "DENSE", 0,
        pre_n_pop, post_n_pop,
        learn_post_weight_update_model, {}, {"w": float_min}, {}, {"s": float_min},
        "DeltaCurr", {}, {})

    # Build model and load
    model.build()
    model.load()

if __name__ == '__main__':
    test_wu_vars_post("single_threaded_cpu", types.Float, False)
