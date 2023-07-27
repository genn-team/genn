import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel
from pygenn.genn import VarAccess, VarAccessMode

from pygenn import (create_current_source_model, 
                    create_custom_update_model,
                    create_neuron_model,
                    create_postsynaptic_model,
                    create_weight_update_model,
                    create_var_ref,
                    init_sparse_connectivity)

neuron_model = create_neuron_model(
    "neuron",
    var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE), ("XShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)])

set_time_custom_update_model = create_custom_update_model(
    "set_time_custom_update",
     update_code=
     """
     V = t;
     R = t;
     """,
     var_name_types=[("V", "scalar")],
     var_refs=[("R", "scalar", VarAccessMode.READ_WRITE)])

set_time_shared_custom_update_model = create_custom_update_model(
    "set_time_custom_update",
     update_code=
     """
     R = t;
     """,
     var_refs=[("R", "scalar", VarAccessMode.READ_WRITE)])
 
softmax_1_custom_update_model = create_custom_update_model(
    "softmax_1",
    update_code=
    """
    MaxY = Y;
    """,
    var_name_types=[("Max", "scalar", VarAccess.REDUCE_NEURON_MAX)],
    var_refs=[("Y", "scalar", VarAccessMode.READ_ONLY)])

softmax_2_custom_update_model = create_custom_update_model(
    "softmax_2",
    update_code=
    """
    SumExpPi = exp(Y - MaxY);
    """,
    var_name_types=[("SumExpPi", "scalar", VarAccess.REDUCE_NEURON_SUM)],
    var_refs=[("Y", "scalar", VarAccessMode.READ_ONLY),
              ("MaxY", "scalar", VarAccessMode.READ_ONLY)])

softmax_3_custom_update_model = create_custom_update_model(
    "softmax_3",
    update_code=
    """
    Pi = exp(Y - MaxY) / SumExpPi;
    """,
    var_refs=[("Y", "scalar", VarAccessMode.READ_ONLY),
              ("MaxY", "scalar", VarAccessMode.READ_ONLY),
              ("SumExpPi", "scalar", VarAccessMode.READ_ONLY),
              ("Pi", "scalar", VarAccessMode.READ_ONLY)])

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_update(backend, precision):
    model = GeNNModel(precision, "test_custom_update", backend=backend)
    model.dt = 1.0
    
    # Create a variety of models to attach custom updates to
    n_pop = model.add_neuron_population("Neurons", 100, neuron_model, 
                                        {}, {"X": 0.0, "XShared": 0.0})
    
    
    # Create set time custom updates
    cu_n = model.add_custom_update("NeuronSetTime", "Test", set_time_custom_update_model,
                                   {}, {"V": 0.0}, {"R": create_var_ref(n_pop, "X")})
    cu_n_shared = model.add_custom_update("NeuronSharedSetTime", "Test", set_time_shared_custom_update_model,
                                          {}, {}, {"R": create_var_ref(n_pop, "XShared")})
                                   
    
    # Build model and load
    model.build()
    model.load()
    
    # Simulate 20 timesteps
    samples = [
        (n_pop, "X", (100,)),
        (cu_n, "V", (100,)),
        (n_pop, "XShared", (1,))]
    while model.timestep < 20:
        # Every 10 timesteps, trigger custom update
        if (model.timestep % 10) == 0:
            model.custom_update("Test")
        model.step_time()

        # Loop through populations
        correct = 10 * ((model.timestep - 1) // 10)
        for pop, var, shape in samples:
            # Pull variable from device
            pop.pull_var_from_device(var)
            
            # If shape of view doesn't match, give error
            view = pop.vars[var].view
            if view.shape != shape:
                assert False, f"{pop.name} var {var} has wrong shape ({view.shape} rather than {shapse})"
            # If values don't match, give error
            elif not np.all(np.isclose(view, correct)):
                assert False, f"{pop.name} var {var} has wrong value ({view} rather than {correct})"
    
@pytest.mark.parametrize("backend", ["cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_update_batch(backend, precision):
    pass


@pytest.mark.parametrize("backend", ["cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_update_batch(backend, precision):
    pass


if __name__ == '__main__':
    test_custom_update("single_threaded_cpu", types.Float)
