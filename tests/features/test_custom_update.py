import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel
from pygenn.genn import CustomUpdateVarAccess, VarAccess, VarAccessMode

from scipy.special import softmax
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
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_toeplitz_connectivity,
                    init_weight_update, init_var)


@pytest.mark.parametrize("backend, batch_size", [("single_threaded_cpu", 1), 
                                                 ("cuda", 1), ("cuda", 5)])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_update(backend, precision, batch_size):
    neuron_model = create_neuron_model(
        "neuron",
        var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE),
                        ("XShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)])

    current_source_model = create_current_source_model(
        "current_source",
        var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE),
                        ("XShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)])

    weight_update_model = create_weight_update_model(
        "weight_update",
        var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE)],
        pre_var_name_types=[("preX", "scalar", VarAccess.READ_ONLY_DUPLICATE),
                            ("preXShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)],
        post_var_name_types=[("postX", "scalar", VarAccess.READ_ONLY_DUPLICATE),
                             ("postXShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)])

    postsynaptic_update_model = create_postsynaptic_model(
        "postsynaptic_update",
        var_name_types=[("psmX", "scalar", VarAccess.READ_ONLY_DUPLICATE),
                        ("psmXShared", "scalar", VarAccess.READ_ONLY_SHARED_NEURON)])

    custom_update_model = create_custom_update_model(
        "custom_update",
        var_name_types=[("X", "scalar", CustomUpdateVarAccess.READ_ONLY)],
        var_refs=[("R", "scalar")])

    set_time_custom_update_model = create_custom_update_model(
        "set_time_custom_update",
         update_code=
         """
         V = (batch * 1000.0) + t;
         R = (batch * 1000.0) + t;
         """,
         var_name_types=[("V", "scalar")],
         var_refs=[("R", "scalar", VarAccessMode.READ_WRITE)])

    set_time_shared_custom_update_model = create_custom_update_model(
        "set_time_custom_update",
         update_code=
         """
         R = (batch * 1000.0) + t;
         """,
         var_refs=[("R", "scalar", VarAccessMode.READ_WRITE)])
 
    model = GeNNModel(precision, "test_custom_update", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size
    
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
    
    conv_params = {"conv_kh": 3, "conv_kw": 3,
                   "conv_ih": 10, "conv_iw": 10, "conv_ic": 1,
                   "conv_oh": 10, "conv_ow": 10, "conv_oc": 1}
    kernel_s_pop = model.add_synapse_population(
        "ToeplitzSynapses", "TOEPLITZ", 0,
        ss_pop, n_pop,
        weight_update_model, {}, {"X": 0.0}, {"preX": 0.0, "preXShared": 0.0}, {"postX": 0.0, "postXShared": 0.0},
        "DeltaCurr", {}, {},
        init_toeplitz_connectivity("Conv2D", conv_params))
    
    cu = model.add_custom_update(
        "CustomUpdate", "Test", custom_update_model,
         {}, {"X": 0.0}, {"R": create_var_ref(n_pop, "X")})
    dense_cu = model.add_custom_update(
        "DenseCustomUpdate", "Test", custom_update_model,
         {}, {"X": 0.0}, {"R": create_wu_var_ref(dense_s_pop, "X")})
    sparse_cu = model.add_custom_update(
        "SparseCustomUpdate", "Test", custom_update_model,
         {}, {"X": 0.0}, {"R": create_wu_var_ref(sparse_s_pop, "X")})
 
    # Create set time custom updates
    set_time_n = model.add_custom_update("NeuronSetTime", "Test", set_time_custom_update_model,
                                         {}, {"V": 0.0}, {"R": create_var_ref(n_pop, "X")})
    model.add_custom_update("NeuronSharedSetTime", "Test", set_time_shared_custom_update_model,
                            {}, {}, {"R": create_var_ref(n_pop, "XShared")})
    set_time_cs = model.add_custom_update("CurrentSourceSetTime", "Test", set_time_custom_update_model,
                                          {}, {"V": 0.0}, {"R": create_var_ref(cs, "X")})
    model.add_custom_update("CurrentSourceSharedSetTime", "Test", set_time_shared_custom_update_model,
                            {}, {}, {"R": create_var_ref(cs, "XShared")})
    set_time_psm_dense = model.add_custom_update("PSMDenseSetTime", "Test", set_time_custom_update_model,
                                                 {}, {"V": 0.0}, {"R": create_psm_var_ref(dense_s_pop, "psmX")})
    model.add_custom_update("PSMDenseSharedSetTime", "Test", set_time_shared_custom_update_model,
                            {}, {}, {"R": create_psm_var_ref(dense_s_pop, "psmXShared")})
    set_time_wu_pre_dense = model.add_custom_update("WUPreDenseSetTime", "Test", set_time_custom_update_model,
                                                    {}, {"V": 0.0}, {"R": create_wu_pre_var_ref(dense_s_pop, "preX")})
    model.add_custom_update("WUPreDenseSharedSetTime", "Test", set_time_shared_custom_update_model,
                            {}, {}, {"R": create_wu_pre_var_ref(dense_s_pop, "preXShared")})
    set_time_wu_post_dense = model.add_custom_update("WUPostDenseSetTime", "Test", set_time_custom_update_model,
                                                     {}, {"V": 0.0}, {"R": create_wu_post_var_ref(dense_s_pop, "postX")})
    model.add_custom_update("WUPostDenseSharedSetTime", "Test", set_time_shared_custom_update_model,
                            {}, {}, {"R": create_wu_post_var_ref(dense_s_pop, "postXShared")})
    set_time_cu = model.add_custom_update("CUSetTime", "Test", set_time_custom_update_model,
                                          {}, {"V": 0.0}, {"R": create_var_ref(cu, "X")})

    # Create set time custom updates on synapse variables
    set_time_wu_dense = model.add_custom_update("WUDenseSetTime", "Test", set_time_custom_update_model,
                                                {}, {"V": 0.0}, {"R": create_wu_var_ref(dense_s_pop, "X")})
    set_time_wu_sparse = model.add_custom_update("WUSparseSetTime", "Test", set_time_custom_update_model,
                                                 {}, {"V": 0.0}, {"R": create_wu_var_ref(sparse_s_pop, "X")})
    set_time_wu_kernel = model.add_custom_update("WUKernelSetTime", "Test", set_time_custom_update_model,
                                                 {}, {"V": 0.0}, {"R": create_wu_var_ref(kernel_s_pop, "X")})
    set_time_cu_dense = model.add_custom_update("CUDenseSetTime", "Test", set_time_custom_update_model,
                                                {}, {"V": 0.0}, {"R": create_wu_var_ref(dense_cu, "X")})
    set_time_cu_sparse = model.add_custom_update("CUSparseSetTime", "Test", set_time_custom_update_model,
                                                 {}, {"V": 0.0}, {"R": create_wu_var_ref(sparse_cu, "X")})

    # Build model and load
    model.build()
    model.load()
    
    # Simulate 20 timesteps
    samples = [
        (n_pop, n_pop.vars["X"], (100,)),
        (set_time_n, set_time_n.vars["V"], (100,)),
        (n_pop, n_pop.vars["XShared"], (1,)),
        (cs, cs.vars["X"], (100,)),
        (set_time_cs, set_time_cs.vars["V"], (100,)),
        (cs, cs.vars["XShared"], (1,)),
        (dense_s_pop, dense_s_pop.psm_vars["psmX"], (100,)),
        (set_time_psm_dense, set_time_psm_dense.vars["V"], (100,)),
        (dense_s_pop, dense_s_pop.psm_vars["psmXShared"], (1,)),
        (dense_s_pop, dense_s_pop.pre_vars["preX"], (10,)),
        (set_time_wu_pre_dense, set_time_wu_pre_dense.vars["V"], (10,)),
        (dense_s_pop, dense_s_pop.pre_vars["preXShared"], (1,)),
        (dense_s_pop, dense_s_pop.post_vars["postX"], (100,)),
        (set_time_wu_post_dense, set_time_wu_post_dense.vars["V"], (100,)),
        (dense_s_pop, dense_s_pop.post_vars["postXShared"], (1,)),
        (cu, cu.vars["X"], (100,)),
        (set_time_cu, set_time_cu.vars["V"], (100,)),
        (dense_s_pop, dense_s_pop.vars["X"], (10 * 100,)),
        (set_time_wu_dense, set_time_wu_dense.vars["V"], (10 * 100,)),
        (sparse_s_pop, sparse_s_pop.vars["X"], (10 * 10,)),
        (set_time_wu_sparse, set_time_wu_sparse.vars["V"], (10 * 10,)),
        (kernel_s_pop, kernel_s_pop.vars["X"], (3 * 3,)),
        (set_time_wu_kernel, set_time_wu_kernel.vars["V"], (3 * 3,)),
        (dense_cu, dense_cu.vars["X"], (10 * 100,)),
        (set_time_cu_dense, set_time_cu_dense.vars["V"], (10 * 100,)),
        (sparse_cu, sparse_cu.vars["X"], (10 * 10,)),
        (set_time_cu_sparse, set_time_cu_sparse.vars["V"], (10 * 10,))]
    while model.timestep < 20:
        # Every 10 timesteps, trigger custom update
        if (model.timestep % 10) == 0:
            model.custom_update("Test")
        model.step_time()

        # Loop through populations
        correct = [(1000 * b)  + (10 * ((model.timestep - 1) // 10)) 
                   for b in range(batch_size)]
        correct = np.reshape(correct, (batch_size, 1))
        for pop, var, shape in samples:
            # Pull variable from device
            var.pull_from_device()
            
            # Add batch size axis to shape
            if batch_size != 1:
                shape = (batch_size,) + shape

            # If shape of view doesn't match, give error
            view = var.view
            if view.shape != shape:
                assert False, f"{pop.name} var {var.name} has wrong shape ({view.shape} rather than {shape})"
            # If values don't match, give error
            elif not np.allclose(view, correct):
                assert False, f"{pop.name} var {var.name} has wrong value ({view} rather than {correct})"

@pytest.mark.parametrize("backend, batch_size", [("single_threaded_cpu", 1), 
                                                 ("cuda", 1), ("cuda", 5)])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_update_transpose(backend, precision, batch_size):
    static_pulse_duplicate_model = create_weight_update_model(
        "static_pulse_duplicate",
        var_name_types=[("g", "scalar", VarAccess.READ_ONLY_DUPLICATE)],
        sim_code=
        """
        addToPost(g);
        """)


    model = GeNNModel(precision, "test_custom_update_transpose", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size
    
    # Create pre and postsynaptic populations
    pre_n_pop = model.add_neuron_population("PreNeurons", 100, "SpikeSource", {}, {}); 
    post_n_pop = model.add_neuron_population("PostNeurons", 100, "SpikeSource", {}, {}); 
    
    # Create forward and transpose synapse populations between populations
    g = (np.random.normal(size=(batch_size, 100 * 100)) if batch_size > 1 
         else np.random.normal(size=(100 * 100)))
    forward_s_pop = model.add_synapse_population(
        "ForwardSynapses", "DENSE", 0,
        pre_n_pop, post_n_pop,
        static_pulse_duplicate_model, {}, {"g": g}, {}, {},
        "DeltaCurr", {}, {})
    transpose_s_pop = model.add_synapse_population(
        "TransposeSynapses", "DENSE", 0,
        post_n_pop, pre_n_pop,
        static_pulse_duplicate_model, {}, {"g": 0.0}, {}, {},
        "DeltaCurr", {}, {})
    
    # Create custom update to calculate transpose
    transpose_cu = model.add_custom_update(
        "Transpose", "Transpose", "Transpose",
        {}, {}, {"variable": create_wu_var_ref(forward_s_pop, "g", transpose_s_pop, "g")})
    
    # Build model and load
    model.build()
    model.load()
    
    # Run custom update to calculate transpose
    model.custom_update("Transpose")
    
    # Pull forward and transpose weights from device
    forward_s_pop.vars["g"].pull_from_device()
    transpose_s_pop.vars["g"].pull_from_device()
    
    # Reshape matrices to square and check transpose
    forward_g = np.reshape(forward_s_pop.vars["g"].view, (batch_size, 100, 100))
    transpose_g = np.reshape(transpose_s_pop.vars["g"].view, (batch_size, 100, 100))
    assert np.allclose(forward_g, np.transpose(transpose_g, axes=(0, 2, 1)))

@pytest.mark.parametrize("backend, batch_size", [("single_threaded_cpu", 1), 
                                                 ("cuda", 1), ("cuda", 5)])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_update_neuron_reduce(backend, precision, batch_size):
    reduction_neuron_model = create_neuron_model(
        "reduction_neuron",
        var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE),
                        ("Y", "scalar", VarAccess.READ_ONLY_DUPLICATE)])

    softmax_1_custom_update_model = create_custom_update_model(
        "softmax_1",
        update_code=
        """
        MaxX = X;
        """,
        var_name_types=[("MaxX", "scalar", CustomUpdateVarAccess.REDUCE_NEURON_MAX)],
        var_refs=[("X", "scalar", VarAccessMode.READ_ONLY)])

    softmax_2_custom_update_model = create_custom_update_model(
        "softmax_2",
        update_code=
        """
        SumExpX = exp(X - MaxX);
        """,
        var_name_types=[("SumExpX", "scalar", CustomUpdateVarAccess.REDUCE_NEURON_SUM)],
        var_refs=[("X", "scalar", VarAccessMode.READ_ONLY),
                  ("MaxX", "scalar", VarAccessMode.READ_ONLY)])

    softmax_3_custom_update_model = create_custom_update_model(
        "softmax_3",
        update_code=
        """
        Y = exp(X - MaxX) / SumExpX;
        """,
        var_refs=[("X", "scalar", VarAccessMode.READ_ONLY),
                  ("MaxX", "scalar", VarAccessMode.READ_ONLY),
                  ("SumExpX", "scalar", VarAccessMode.READ_ONLY),
                  ("Y", "scalar", VarAccessMode.READ_WRITE)])
              
    model = GeNNModel(precision, "test_custom_update_neuron_reduce", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size

    # Create a neuron model with two state variables
    x = (np.random.uniform(high=100.0, size=(batch_size, 50)) if batch_size > 1
         else np.random.uniform(high=100.0, size=(50)))
    n_pop = model.add_neuron_population("Neurons", 50, reduction_neuron_model, 
                                        {}, {"X": x, "Y": 0.0})

    # Create softmax custom update
    softmax_1_cu = model.add_custom_update("Softmax1", "Softmax1", softmax_1_custom_update_model,
                                           {}, {"MaxX": 0.0}, {"X": create_var_ref(n_pop, "X")})
    softmax_2_cu = model.add_custom_update("Softmax2", "Softmax2", softmax_2_custom_update_model,
                                           {}, {"SumExpX": 0.0}, {"X": create_var_ref(n_pop, "X"),
                                                                  "MaxX": create_var_ref(softmax_1_cu, "MaxX")})
    model.add_custom_update("Softmax3", "Softmax3", softmax_3_custom_update_model,
                            {}, {}, {"X": create_var_ref(n_pop, "X"),
                                    "MaxX": create_var_ref(softmax_1_cu, "MaxX"),
                                    "SumExpX": create_var_ref(softmax_2_cu, "SumExpX"),
                                    "Y": create_var_ref(n_pop, "Y")})

    # Build model and load
    model.build()
    model.load()

    # Launch sequence of softmax update
    model.custom_update("Softmax1")
    model.custom_update("Softmax2")
    model.custom_update("Softmax3")

    # Download X and Y 
    n_pop.vars["Y"].pull_from_device()

    # Compare Y to softmax calculated with SciPy
    if batch_size == 1:
        assert np.allclose(softmax(x), n_pop.vars["Y"].view)
    else:
        assert np.allclose(softmax(x, axis=1), n_pop.vars["Y"].view)


@pytest.mark.parametrize("backend, batch_size", [("single_threaded_cpu", 1), 
                                                 ("cuda", 1), ("cuda", 5)])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_update_batch_reduction(backend, precision, batch_size):
    # **TODO** once VarAccess is refactored, we should really be able to reduce neuron shared across batch dimension
    neuron_model = create_neuron_model(
        "neuron",
        var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE),
                        ("SumX", "scalar", VarAccess.READ_ONLY)])

    weight_update_model = create_weight_update_model(
        "weight_update",
        var_name_types=[("X", "scalar", VarAccess.READ_ONLY_DUPLICATE),
                        ("SumX", "scalar", VarAccess.READ_ONLY)])
   
    reduction_custom_update_model = create_custom_update_model(
        "reduction_custom_update",
        update_code=
        """
        SumX = X;
        MaxX = X;
        """,
        var_name_types=[("MaxX", "scalar", CustomUpdateVarAccess.REDUCE_BATCH_MAX)],
        var_refs=[("X", "scalar", VarAccessMode.READ_ONLY),
                  ("SumX", "scalar", VarAccessMode.REDUCE_SUM)])

    model = GeNNModel(precision, "test_custom_update_batch_reduction", 
                      backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size
    
    # Create a variety of models to attach custom updates to
    ss_pop = model.add_neuron_population("SpikeSource", 10, "SpikeSource", {}, {});
    
    x_n = (np.random.uniform(high=100.0, size=(batch_size, 100)) if batch_size > 1
           else np.random.uniform(high=100.0, size=100))
    n_pop = model.add_neuron_population("Neurons", 100, neuron_model, 
                                        {}, {"X": x_n, "SumX": 0.0})
    
    x_dense_s = (np.random.uniform(high=100.0, size=(batch_size, 1000)) if batch_size > 1
                 else np.random.uniform(high=100.0, size=1000))
    dense_s_pop = model.add_synapse_population(
        "DenseSynapses", "DENSE", 0,
        ss_pop, n_pop,
        weight_update_model, {}, {"X": x_dense_s, "SumX": 0.0}, {}, {},
        "DeltaCurr", {}, {})

    x_sparse_s = (np.random.uniform(high=100.0, size=(batch_size, 100)) if batch_size > 1
                 else np.random.uniform(high=100.0, size=100))
    pre_ind_sparse = np.repeat(np.arange(10), 10)
    post_ind_sparse = np.random.randint(0, 100, len(pre_ind_sparse))
    sparse_s_pop = model.add_synapse_population(
        "SparseSynapses", "SPARSE", 0,
        ss_pop, n_pop,
        weight_update_model, {}, {"X": x_sparse_s, "SumX": 0.0}, {}, {},
        "DeltaCurr", {}, {})
    sparse_s_pop.set_sparse_connections(pre_ind_sparse, post_ind_sparse)
    
    x_kern_s = (np.random.uniform(high=100.0, size=(batch_size, 9)) if batch_size > 1
                 else np.random.uniform(high=100.0, size=9))
    conv_params = {"conv_kh": 3, "conv_kw": 3,
                   "conv_ih": 10, "conv_iw": 10, "conv_ic": 1,
                   "conv_oh": 10, "conv_ow": 10, "conv_oc": 1}
    kern_s_pop = model.add_synapse_population(
        "ToeplitzSynapses", "TOEPLITZ", 0,
        ss_pop, n_pop,
        weight_update_model, {}, {"X": x_kern_s, "SumX": 0.0}, {}, {},
        "DeltaCurr", {}, {},
        init_toeplitz_connectivity("Conv2D", conv_params))

    # Create reduction custom updates
    reduce_n = model.add_custom_update("NeuronReduce", "Test", reduction_custom_update_model,
                                       {}, {"MaxX": 0.0}, {"X": create_var_ref(n_pop, "X"), "SumX": create_var_ref(n_pop, "SumX")})
    reduce_s_dense = model.add_custom_update("DenseSynapseReduce", "Test", reduction_custom_update_model,
                                             {}, {"MaxX": 0.0}, {"X": create_wu_var_ref(dense_s_pop, "X"), "SumX": create_wu_var_ref(dense_s_pop, "SumX")})
    reduce_s_sparse = model.add_custom_update("SparseSynapseReduce", "Test", reduction_custom_update_model,
                                              {}, {"MaxX": 0.0}, {"X": create_wu_var_ref(sparse_s_pop, "X"), "SumX": create_wu_var_ref(sparse_s_pop, "SumX")})
    reduce_s_kernel = model.add_custom_update("KernelSynapseReduce", "Test", reduction_custom_update_model,
                                              {}, {"MaxX": 0.0}, {"X": create_wu_var_ref(kern_s_pop, "X"), "SumX": create_wu_var_ref(kern_s_pop, "SumX")})

    # Build model and load
    model.build()
    model.load()

    # Launch custom update to perform reductions
    model.custom_update("Test")

    # Check neuron reduction
    n_pop.vars["SumX"].pull_from_device()
    reduce_n.vars["MaxX"].pull_from_device()
    assert np.allclose(np.sum(x_n, axis=0) if batch_size > 1 else x_n,
                       n_pop.vars["SumX"].view)
    assert np.allclose(np.max(x_n, axis=0) if batch_size > 1 else x_n,
                       reduce_n.vars["MaxX"].view)

    # Loop through synapse reductions
    x_sparse_s_order = (x_sparse_s[:,sparse_s_pop.synapse_order] if batch_size > 1
                        else x_sparse_s[sparse_s_pop.synapse_order])
    samples = [
        (x_dense_s, dense_s_pop, reduce_s_dense),
        (x_sparse_s_order, sparse_s_pop, reduce_s_sparse),
        (x_kern_s, kern_s_pop, reduce_s_kernel)]
    for data, sum_pop, max_pop in samples:
        # Check sum
        sum_pop.pull_var_from_device("SumX")
        sum_correct = np.sum(data, axis=0) if batch_size > 1 else data
        sum_value = sum_pop.get_var_values("SumX") if batch_size > 1 else data
        if not np.allclose(sum_correct, sum_value):
            assert False, f"{sum_pop.name} var SumX has wrong value ({sum_value} rather than {sum_correct})"

        # Check max
        max_pop.pull_var_from_device("MaxX")
        max_correct = np.max(data, axis=0) if batch_size > 1 else data
        max_value = max_pop.get_var_values("MaxX")
        if not np.allclose(max_correct, max_value):
            assert False, f"{max_pop.name} var MaxX has wrong value ({max_value} rather than {max_correct})"

if __name__ == '__main__':
    test_custom_update("cuda", types.Float, 5)
    
