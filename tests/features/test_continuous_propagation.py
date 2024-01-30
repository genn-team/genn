import numpy as np
import pytest
from pygenn import types

from pygenn.genn import ParallelismHint, VarAccess, VarAccessMode
from pygenn import (create_neuron_model,
                    create_sparse_connect_init_snippet,
                    create_var_init_snippet, create_var_ref,
                    create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity, 
                    init_toeplitz_connectivity,
                    init_weight_update, init_var)

pre_neuron_model = create_neuron_model(
    "pre_neuron",
    sim_code=
    """
    x = (id == (int)t) ? 1.0 : 0.0;
    """,
    var_name_types=[("x", "scalar")])

post_neuron_model = create_neuron_model(
    "post_neuron",
    sim_code=
    """
    x = Isyn;
    """,
    var_name_types=[("x", "scalar")])

decoder_model = create_sparse_connect_init_snippet(
    "decoder",
    row_build_code=
    """
    for(unsigned int j = 0; j < num_post; j++) {
        const unsigned int jValue = (1 << j);
        if(((id_pre + 1) & jValue) != 0) {
            addSynapse(j);
        }
    }
    """)

decoder_dense_model = create_var_init_snippet(
    "decoder_dense",
    var_init_code=
    """
    const unsigned int jValue = (1 << id_post);
    value = (((id_pre + 1) & jValue) != 0) ? 1.0 : 0.0;
    """)

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_forward(make_model, backend, precision):
    continous_weight_update_model = create_weight_update_model(
        "continous_weight_update",
        var_name_types=[("g", "scalar", VarAccess.READ_ONLY)],
        pre_neuron_var_refs=[("x", "scalar", VarAccessMode.READ_ONLY)],
        synapse_dynamics_code=
        """
        addToPost(g * x);
        """)

    continous_constant_weight_update_model = create_weight_update_model(
        "continous_constant_weight_update",
        param_names=["g"],
        pre_neuron_var_refs=[("x", "scalar", VarAccessMode.READ_ONLY)],
        synapse_dynamics_code=
        """
        addToPost(g * x);
        """)
        
    model = make_model(precision, "test_forward", backend=backend)
    model.dt = 1.0

    # Create neuron model to generate one-hot pattern to decode
    pre_pop = model.add_neuron_population("Pre", 16, pre_neuron_model, {},
                                          {"x": 0.0})

    # Build sparse connectivity
    pre_inds = []
    post_inds = []
    for i in range(16):
        for j in range(4):
            j_value = 1 << j
            if ((i + 1) & j_value) != 0:
                pre_inds.append(i)
                post_inds.append(j)
    pre_inds = np.asarray(pre_inds)
    post_inds = np.asarray(post_inds)

    # Use to build dense matrix
    dense = np.zeros((16, 4))
    dense[pre_inds,post_inds] = 1.0

    # Create one output neuron pop with constant weight sparse decoder population
    sparse_constant_weight_n_pop = model.add_neuron_population(
        "PostSparseConstantWeightNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "SparseConstantWeightSynapse", "SPARSE",
        pre_pop, sparse_constant_weight_n_pop,
        init_weight_update(continous_constant_weight_update_model, {"g": 1.0},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))

    # Create one output neuron pop with constant weight 
    # sparse decoder population and presynaptic parallelism
    sparse_constant_weight_pre_n_pop = model.add_neuron_population(
        "PostSparseConstantWeightPreSpanNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    sparse_constant_weight_pre_s_pop = model.add_synapse_population(
        "SparseConstantWeightPreSpanSynapse", "SPARSE",
        pre_pop, sparse_constant_weight_pre_n_pop,
        init_weight_update(continous_constant_weight_update_model, {"g": 1.0},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))
    sparse_constant_weight_pre_s_pop.parallelism_hint = ParallelismHint.PRESYNAPTIC

    # Create one output neuron pop with constant weight sparse decoder population
    manual_sparse_constant_weight_n_pop = model.add_neuron_population(
        "ManualPostSparseConstantWeightNeuron", 4, post_neuron_model,
        {}, {"x": 0.0})
    manual_sparse_constant_weight_s_pop = model.add_synapse_population(
        "ManualSparseConstantWeightSynapse", "SPARSE",
        pre_pop, manual_sparse_constant_weight_n_pop,
        init_weight_update(continous_constant_weight_update_model, {"g": 1.0},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr"))
    manual_sparse_constant_weight_s_pop.set_sparse_connections(pre_inds,
                                                               post_inds)

    # Create one output neuron pop with sparse decoder population
    sparse_n_pop = model.add_neuron_population(
        "PostSparseNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "SparseSynapse", "SPARSE",
        pre_pop, sparse_n_pop,
        init_weight_update(continous_weight_update_model, vars={"g": 1.0},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))

    # Create one output neuron pop with sparse 
    # decoder population and presynaptic parallelism
    sparse_pre_n_pop = model.add_neuron_population(
        "PostSparsePreSpanNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    sparse_pre_s_pop = model.add_synapse_population(
        "SparsePreSpanSynapse", "SPARSE",
        pre_pop, sparse_pre_n_pop,
        init_weight_update(continous_weight_update_model, vars={"g": 1.0},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))
    sparse_pre_s_pop.parallelism_hint = ParallelismHint.PRESYNAPTIC

    # Create one output neuron pop with sparse 
    # decoder population and presynaptic parallelism
    sparse_hybrid_n_pop = model.add_neuron_population(
        "PostSparseHybridNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    sparse_hybrid_s_pop = model.add_synapse_population(
        "SparseHybridSynapse", "SPARSE",
        pre_pop, sparse_hybrid_n_pop,
 
        init_weight_update(continous_weight_update_model, vars={"g": 1.0},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))
    sparse_hybrid_s_pop.parallelism_hint = ParallelismHint.PRESYNAPTIC
    sparse_hybrid_s_pop.num_threads_per_spike = 2

    # Create one output neuron pop with sparse decoder population
    manual_sparse_n_pop = model.add_neuron_population(
        "ManualPostSparseNeuron", 4, post_neuron_model,
        {}, {"x": 0.0})
    manual_sparse_s_pop = model.add_synapse_population(
        "ManualSparseSynapse", "SPARSE",
        pre_pop, manual_sparse_n_pop,
        init_weight_update(continous_weight_update_model, vars={"g": 1.0},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))
    manual_sparse_s_pop.set_sparse_connections(pre_inds, post_inds)

    # Create one output neuron pop with dense decoder population
    dense_n_pop = model.add_neuron_population(
        "PostDenseNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "PostDenseSynapse", "DENSE",
        pre_pop, dense_n_pop,
        init_weight_update(continous_weight_update_model, vars={"g": init_var(decoder_dense_model, {})},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr"))

    # Create one output neuron pop with dense decoder population
    manual_dense_n_pop = model.add_neuron_population(
        "ManualPostDenseNeuron", 4, post_neuron_model,
        {}, {"x": 0.0})
    model.add_synapse_population(
        "ManualPostDenseSynapse", "DENSE",
        pre_pop, manual_dense_n_pop,
        init_weight_update(continous_weight_update_model, vars={"g": dense.flatten()},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr", {}, {}))

    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    output_place_values = 2 ** np.arange(4)
    output_populations = [sparse_constant_weight_n_pop,
                          sparse_constant_weight_pre_n_pop,
                          manual_sparse_constant_weight_n_pop,
                          sparse_n_pop, sparse_pre_n_pop, sparse_hybrid_n_pop,
                          manual_sparse_n_pop, dense_n_pop, 
                          manual_dense_n_pop]
    while model.timestep < 15:
        model.step_time()

        # Loop through output populations
        for pop in output_populations:
            # Pull state variable
            pop.vars["x"].pull_from_device()

            # Convert to binary mask
            output_binary = np.isclose(np.ones(4), pop.vars["x"].view)
            
            # Sum up active place values
            output_value = np.sum(output_place_values[output_binary])
            if output_value != (model.timestep - 1):
                assert False, f"{pop.name} decoding incorrect ({output_value} rather than {model.timestep - 1})"

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_forward_den_delay(make_model, backend, precision):
    continous_den_delay_wum = create_weight_update_model(
        "continous_den_delay",
        var_name_types=[("g", "scalar", VarAccess.READ_ONLY),
                        ("d", "uint8_t", VarAccess.READ_ONLY)],
        pre_neuron_var_refs=[("x", "scalar", VarAccessMode.READ_ONLY)],
        synapse_dynamics_code=
        """
        addToPostDelay(g * x, d);
        """)

    model = make_model(precision, "test_forward_den_delay", backend=backend)
    model.dt = 1.0

    # Create neuron model to generate one-hot pattern to decode
    pre_pop = model.add_neuron_population("Pre", 10, pre_neuron_model, {},
                                          {"x": 0.0})

    # Create one output neuron pop with dense decoder population
    delay = np.arange(9, -1, -1)
    dense_n_pop = model.add_neuron_population(
        "PostDenseNeuron", 1, post_neuron_model,
        {}, {"x": 0.0})
    dense_s_pop = model.add_synapse_population(
        "PostDenseSynapse", "DENSE",
        pre_pop, dense_n_pop,
        init_weight_update(continous_den_delay_wum, {}, {"g": 1.0, "d": delay},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr", {}, {}))
    dense_s_pop.max_dendritic_delay_timesteps = 10

    # Create one output neuron pop with sparse decoder population
    sparse_n_pop = model.add_neuron_population(
        "PostSparseNeuron", 1, post_neuron_model,
        {}, {"x": 0.0})
    sparse_s_pop = model.add_synapse_population(
        "PostSparseSynapse", "SPARSE",
        pre_pop, sparse_n_pop,
        init_weight_update(continous_den_delay_wum, {}, {"g": 1.0, "d": delay},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr", {}, {}))
    sparse_s_pop.max_dendritic_delay_timesteps = 10
    sparse_s_pop.set_sparse_connections(np.arange(10), np.zeros(10, dtype=int))
    
    # Create one output neuron pop with sparse decoder population and presynaptic parallelism
    sparse_pre_n_pop = model.add_neuron_population(
        "PostSparsePreSpanNeuron", 1, post_neuron_model,
        {}, {"x": 0.0})
    sparse_pre_s_pop = model.add_synapse_population(
        "PostSparsePreSpanSynapse", "SPARSE",
        pre_pop, sparse_pre_n_pop, 
        init_weight_update(continous_den_delay_wum, {}, {"g": 1.0, "d": delay},
                           pre_var_refs={"x": create_var_ref(pre_pop, "x")}),
        init_postsynaptic("DeltaCurr", {}, {}))
    sparse_pre_s_pop.max_dendritic_delay_timesteps = 10
    sparse_pre_s_pop.set_sparse_connections(np.arange(10), np.zeros(10, dtype=int))
    sparse_pre_s_pop.parallelism_hint = ParallelismHint.PRESYNAPTIC

    # Build model and load
    model.build()
    model.load()

    # Simulate for 11 timesteps
    output_populations = [dense_n_pop, sparse_n_pop, sparse_pre_n_pop]
    while model.timestep < 11:
        model.step_time()

        # Loop through output populations
        correct = 10.0 if model.timestep == 11 else 0.0
        for pop in output_populations:
            # Pull state variable
            pop.vars["x"].pull_from_device()

            # If not close to correct value, error
            if not np.isclose(pop.vars["x"].view[0], correct):
                assert False, f"{pop.name} decoding incorrect ({pop.vars['x'].view[0]} rather than {correct})"


@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_reverse(make_model, backend, precision):
    pre_reverse_neuron_model = create_neuron_model(
        "pre_reverse_neuron",
        sim_code=
        """
        y = Isyn;
        x = (id == (int)t) ? 1.0 : 0.0;
        """,
        var_name_types=[("x", "scalar"), ("y", "scalar")])

    continous_reverse_model = create_weight_update_model(
        "continous_reverse",
        var_name_types=[("g", "scalar", VarAccess.READ_ONLY)],
        pre_neuron_var_refs=[("x", "scalar", VarAccessMode.READ_ONLY)],
        synapse_dynamics_code=
        """
        addToPre(g * x);
        """)

    model = make_model(precision, "test_reverse", backend=backend)
    model.dt = 1.0

    # Create neuron model with extra y variable 
    # to generate one-hot pattern to decode
    pre_n_pop = model.add_neuron_population(
        "SpikeSource", 16, pre_reverse_neuron_model,
        {}, {"x": 0.0, "y": 0.0})
    pre_pre_n_pop = model.add_neuron_population(
        "PreSpikeSource", 16, pre_reverse_neuron_model,
        {}, {"x": 0.0, "y": 0.0})

    # Add postsynptic population to connect to
    post_n_pop = model.add_neuron_population(
        "Post", 4, post_neuron_model,
        {}, {"x": 0.0})

    # Build sparse connectivity
    pre_inds = []
    post_inds = []
    weights = []
    for i in range(16):
        for j in range(4):
            j_value = 1 << j
            if ((i + 1) & j_value) != 0:
                pre_inds.append(i)
                post_inds.append(j)
                weights.append(float(j_value))

    pre_inds = np.asarray(pre_inds)
    post_inds = np.asarray(post_inds)
    weights = np.asarray(weights)

    # Add connectivity
    s_pop = model.add_synapse_population(
        "SparseSynapse", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(continous_reverse_model, {}, {"g": weights},
                           pre_var_refs={"x": create_var_ref(pre_n_pop, "x")}),
        init_postsynaptic("DeltaCurr"))
    s_pop.set_sparse_connections(pre_inds, post_inds)

    s_pre_pop = model.add_synapse_population(
        "SparsePreSynapse", "SPARSE",
        pre_pre_n_pop, post_n_pop,
        init_weight_update(continous_reverse_model, {}, {"g": weights},
                           pre_var_refs={"x": create_var_ref(pre_pre_n_pop, "x")}),
        init_postsynaptic("DeltaCurr"))
    s_pre_pop.set_sparse_connections(pre_inds, post_inds)
    s_pre_pop.parallelism_hint = ParallelismHint.PRESYNAPTIC

    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    while model.timestep < 16:
        model.step_time()

        pre_n_pop.vars["y"].pull_from_device()
        pre_pre_n_pop.vars["y"].pull_from_device()

        assert np.sum(pre_n_pop.vars["y"].view) == (model.timestep - 1)
        assert np.sum(pre_pre_n_pop.vars["y"].view) == (model.timestep - 1)
