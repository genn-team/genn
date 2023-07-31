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
                    create_psm_var_ref,
                    create_wu_var_ref,
                    create_wu_pre_var_ref,
                    create_wu_post_var_ref,
                    init_sparse_connectivity)

neuron_model = create_neuron_model(
    "neuron",
    var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE), ("XShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)])

current_source_model = create_current_source_model(
    "current_source",
    var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE), ("XShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)])

weight_update_model = create_weight_update_model(
    "weight_update",
    var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE)],
    pre_var_name_types=[("preX", "scalar", VarAccess.READ_ONLY_DUPLICATE), ("preXShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)],
    post_var_name_types=[("postX", "scalar", VarAccess.READ_ONLY_DUPLICATE), ("postXShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)])

postsynaptic_update_model = create_postsynaptic_model(
    "postsynaptic_update",
    var_name_types=[("psmX", "scalar", VarAccess.READ_ONLY_DUPLICATE), ("psmXShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)])

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
    ss_pop = model.add_neuron_population("SpikeSource", 10, "SpikeSource", {}, {});
    n_pop = model.add_neuron_population("Neurons", 100, neuron_model, 
                                        {}, {"X": 0.0, "XShared": 0.0})
    cs = model.add_current_source("CurrentSource", current_source_model, n_pop,
                                  {}, {"X": 0.0, "XShared": 0.0})
    
    dense_s_pop = model.add_synapse_population(
        "DenseSynapses", "DENSE", 0,
        ss_pop, n_pop,
        weight_update_model, {}, {"X": 0.0}, {"preX": 0.0, "preXShared": 0.0}, {"postX": 0.0, "postXShared": 0.0},
        postsynaptic_update_model, {}, {"psmX": 0.0, "psmXShared": 0.0})
    sparse_s_pop = model.add_synapse_population(
        "SparseSynapses", "SPARSE", 0,
        ss_pop, n_pop,
        weight_update_model, {}, {"X": 0.0}, {"preX": 0.0, "preXShared": 0.0}, {"postX": 0.0, "postXShared": 0.0},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("FixedNumberPostWithReplacement", {"rowLength": 10}))
        
    # Create set time custom updates
    cu_n = model.add_custom_update("NeuronSetTime", "Test", set_time_custom_update_model,
                                   {}, {"V": 0.0}, {"R": create_var_ref(n_pop, "X")})
    model.add_custom_update("NeuronSharedSetTime", "Test", set_time_shared_custom_update_model,
                            {}, {}, {"R": create_var_ref(n_pop, "XShared")})
    cu_cs = model.add_custom_update("CurrentSourceSetTime", "Test", set_time_custom_update_model,
                                    {}, {"V": 0.0}, {"R": create_var_ref(cs, "X")})
    model.add_custom_update("CurrentSourceSharedSetTime", "Test", set_time_shared_custom_update_model,
                            {}, {}, {"R": create_var_ref(cs, "XShared")})
    cu_psm_dense = model.add_custom_update("PSMDenseSetTime", "Test", set_time_custom_update_model,
                                           {}, {"V": 0.0}, {"R": create_psm_var_ref(dense_s_pop, "psmX")})
    model.add_custom_update("PSMDenseSharedSetTime", "Test", set_time_shared_custom_update_model,
                            {}, {}, {"R": create_psm_var_ref(dense_s_pop, "psmXShared")})
    cu_wu_pre_dense = model.add_custom_update("WUPreDenseSetTime", "Test", set_time_custom_update_model,
                                              {}, {"V": 0.0}, {"R": create_wu_pre_var_ref(dense_s_pop, "preX")})
    model.add_custom_update("WUPreDenseSharedSetTime", "Test", set_time_shared_custom_update_model,
                            {}, {}, {"R": create_wu_pre_var_ref(dense_s_pop, "preXShared")})
    cu_wu_post_dense = model.add_custom_update("WUPostDenseSetTime", "Test", set_time_custom_update_model,
                                               {}, {"V": 0.0}, {"R": create_wu_post_var_ref(dense_s_pop, "postX")})
    model.add_custom_update("WUPostDenseSharedSetTime", "Test", set_time_shared_custom_update_model,
                            {}, {}, {"R": create_wu_post_var_ref(dense_s_pop, "postXShared")})

    # Create set time custom updates on synapse variables
    cu_wu_dense = model.add_custom_update("WUDenseSetTime", "Test", set_time_custom_update_model,
                                          {}, {"V": 0.0}, {"R": create_wu_var_ref(dense_s_pop, "X")})
    cu_wu_sparse = model.add_custom_update("WUSparseSetTime", "Test", set_time_custom_update_model,
                                           {}, {"V": 0.0}, {"R": create_wu_var_ref(sparse_s_pop, "X")})
                                          
    # Build model and load
    model.build()
    model.load()
    
    # Simulate 20 timesteps
    samples = [
        (n_pop, "X", n_pop.vars, (100,)),
        (cu_n, "V", cu_n.vars, (100,)),
        (n_pop, "XShared", n_pop.vars, (1,)),
        (cs, "X", cs.vars, (100,)),
        (cu_cs, "V", cu_cs.vars, (100,)),
        (cs, "XShared", cs.vars, (1,)),
        (dense_s_pop, "psmX", dense_s_pop.psm_vars, (100,)),
        (cu_psm_dense, "V", cu_psm_dense.vars, (100,)),
        (dense_s_pop, "psmXShared", dense_s_pop.psm_vars, (1,)),
        (dense_s_pop, "preX", dense_s_pop.pre_vars, (10,)),
        (cu_wu_pre_dense, "V", cu_wu_pre_dense.vars, (10,)),
        (dense_s_pop, "preXShared", dense_s_pop.pre_vars, (1,)),
        (dense_s_pop, "postX", dense_s_pop.post_vars, (100,)),
        (cu_wu_post_dense, "V", cu_wu_post_dense.vars, (100,)),
        (dense_s_pop, "postXShared", dense_s_pop.post_vars, (1,)),
        (dense_s_pop, "X", dense_s_pop.vars, (10 * 100,)),
        (cu_wu_dense, "V", cu_wu_dense.vars, (10 * 100,)),
        (sparse_s_pop, "X", sparse_s_pop.vars, (10 * 10,)),
        (cu_wu_sparse, "V", cu_wu_sparse.vars, (10 * 10,))]
    while model.timestep < 20:
        # Every 10 timesteps, trigger custom update
        if (model.timestep % 10) == 0:
            model.custom_update("Test")
        model.step_time()

        # Loop through populations
        correct = 10 * ((model.timestep - 1) // 10)
        for pop, var_name, vars, shape in samples:
            # Pull variable from device
            pop.pull_var_from_device(var_name)
            
            # If shape of view doesn't match, give error
            view = vars[var_name].view
            if view.shape != shape:
                assert False, f"{pop.name} var {var_name} has wrong shape ({view.shape} rather than {shape})"
            # If values don't match, give error
            elif not np.all(np.isclose(view, correct)):
                assert False, f"{pop.name} var {var_name} has wrong value ({view} rather than {correct})"

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
