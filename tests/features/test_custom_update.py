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
    var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE), ("XShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON), 
                    ("Y", "scalar", VarAccess.READ_ONLY_DUPLICATE)])

set_time_custom_update_model = create_custom_update_model(
    "set_time_custom_update",
     update_code=
     """
     V = t;
     R = t;
     """,
     var_name_types=[("V", "scalar")],
     var_refs=[("R", "scalar", VarAccessMode.READ_WRITE)])

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_update(backend, precision):
    model = GeNNModel(precision, "test_custom_update", backend=backend)
    model.dt = 1.0
    
    # Create a variety of models to attach custom updates to
    n_pop = model.add_neuron_population("Neurons", 100, neuron_model, 
                                        {}, {"X": 0.0, "XShared": 0.0, "Y": 0.0})
    
    
    # Create set time custom updates
    cu_n = model.add_custom_update("NeuronsSetTime", "Test", set_time_custom_update_model,
                                   {}, {"V": 0.0}, {"R": create_var_ref(n_pop, "X")})
    
    #SetTimeShared::VarReferences neuronSharedVarReferences(createVarRef(ng, "VShared")); // R
    #model.addCustomUpdate<SetTimeShared>("NeuronSharedSetTime", "Test",
    #                                     {}, {}, neuronSharedVarReferences);
    # Build model and load
    model.build()
    model.load()
    
@pytest.mark.parametrize("backend", ["cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_update_batch(backend, precision):
    pass


@pytest.mark.parametrize("backend", ["cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_update_batch(backend, precision):
    pass


if __name__ == '__main__':
    test_custom_update("cuda", types.Float)
