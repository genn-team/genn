import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel

from pygenn.genn import SpanType, VarAccess
from pygenn import (create_neuron_model,
                    create_sparse_connect_init_snippet,
                    create_var_init_snippet,
                    create_weight_update_model,
                    init_sparse_connectivity, init_var)

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

pre_reverse_model = create_neuron_model(
    "pre_reverse",
    var_name_types=[("x", "scalar")],
    sim_code=
    """
    x = Isyn;
    """)
    
pre_reverse_spike_source_model = create_neuron_model(
    "pre_reverse_spike_source",
    var_name_types=[("startSpike", "unsigned int"), 
                    ("endSpike", "unsigned int", VarAccess.READ_ONLY_DUPLICATE),
                    ("x", "scalar")],
    extra_global_params=[("spikeTimes", "scalar*")],
    sim_code=
    """
    x = Isyn;
    """,
    threshold_condition_code=
    """
    startSpike != endSpike && t >= spikeTimes[startSpike]
    """,
    reset_code=
    """
    startSpike++;
    """)

post_neuron_model = create_neuron_model(
    "post_neuron",
    sim_code=
    """
    x = Isyn;
    """,
    var_name_types=[("x", "scalar")])

static_pulse_reverse_model = create_weight_update_model(
    "static_pulse_reverse",
    sim_code=
    """
    $(addToPre, $(g));
    """,
    var_name_types=[("g", "scalar", VarAccess.READ_ONLY)])

static_pulse_reverse_post_model = create_weight_update_model(
    "static_pulse_reverse_post",
    learn_post_code=
    """
    $(addToPre, $(g));
    """,
    var_name_types=[("g", "scalar", VarAccess.READ_ONLY)])

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_forward(backend, precision):
    model = GeNNModel(precision, "test_forward", backend=backend)
    model.dt = 1.0

    # Create spike source array to generate one-hot pattern to decode
    ss_pop = model.add_neuron_population("SpikeSource", 16, "SpikeSourceArray",
                                         {}, {"startSpike": np.arange(16), "endSpike": np.arange(1, 17)})
    ss_pop.extra_global_params["spikeTimes"].set_values(np.arange(16.0))

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
        "SparseConstantWeightSynapse", "SPARSE", 0,
        ss_pop, sparse_constant_weight_n_pop,
        "StaticPulseConstantWeight", {"g": 1.0}, {}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity(decoder_model, {}))
    
    # Create one output neuron pop with constant weight 
    # sparse decoder population and presynaptic parallelism
    sparse_constant_weight_pre_n_pop = model.add_neuron_population(
        "PostSparseConstantWeightPreSpanNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    sparse_constant_weight_pre_s_pop = model.add_synapse_population(
        "SparseConstantWeightPreSpanSynapse", "SPARSE", 0,
        ss_pop, sparse_constant_weight_pre_n_pop,
        "StaticPulseConstantWeight", {"g": 1.0}, {}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity(decoder_model, {}))
    sparse_constant_weight_pre_s_pop.span_type = SpanType.PRESYNAPTIC

    # Create one output neuron pop with constant weight sparse decoder population
    manual_sparse_constant_weight_n_pop = model.add_neuron_population(
        "ManualPostSparseConstantWeightNeuron", 4, post_neuron_model,
        {}, {"x": 0.0})
    manual_sparse_constant_weight_s_pop = model.add_synapse_population(
        "ManualSparseConstantWeightSynapse", "SPARSE", 0,
        ss_pop, manual_sparse_constant_weight_n_pop,
        "StaticPulseConstantWeight", {"g": 1.0}, {}, {}, {},
        "DeltaCurr", {}, {})
    manual_sparse_constant_weight_s_pop.set_sparse_connections(pre_inds,
                                                               post_inds)

    # Create one output neuron pop with sparse decoder population
    sparse_n_pop = model.add_neuron_population(
        "PostSparseNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "SparseSynapse", "SPARSE", 0,
        ss_pop, sparse_n_pop,
        "StaticPulse", {}, {"g": 1.0}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity(decoder_model, {}))
    
    # Create one output neuron pop with sparse 
    # decoder population and presynaptic parallelism
    sparse_pre_n_pop = model.add_neuron_population(
        "PostSparsePreSpanNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    sparse_pre_s_pop = model.add_synapse_population(
        "SparsePreSpanSynapse", "SPARSE", 0,
        ss_pop, sparse_pre_n_pop,
        "StaticPulse", {}, {"g": 1.0}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity(decoder_model, {}))
    sparse_pre_s_pop.span_type = SpanType.PRESYNAPTIC
    
    # Create one output neuron pop with sparse decoder population
    manual_sparse_n_pop = model.add_neuron_population(
        "ManualPostSparseNeuron", 4, post_neuron_model,
        {}, {"x": 0.0})
    manual_sparse_s_pop = model.add_synapse_population(
        "ManualSparseSynapse", "SPARSE", 0,
        ss_pop, manual_sparse_n_pop,
        "StaticPulse", {}, {"g": 1.0}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity(decoder_model, {}))
    manual_sparse_s_pop.set_sparse_connections(pre_inds, post_inds)

    # Create one output neuron pop with bitmask decoder population
    bitmask_n_pop = model.add_neuron_population(
        "PostBitmaskNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "BitmaskSynapse", "SPARSE", 0,
        ss_pop, bitmask_n_pop,
        "StaticPulseConstantWeight", {"g": 1.0}, {}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity(decoder_model, {}))

    # Create one output neuron pop with dense decoder population
    dense_n_pop = model.add_neuron_population(
        "PostDenseNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "PostDenseSynapse", "DENSE", 0,
        ss_pop, dense_n_pop,
        "StaticPulse", {}, {"g": init_var(decoder_dense_model, {})}, {}, {},
        "DeltaCurr", {}, {})

    # Create one output neuron pop with dense decoder population
    manual_dense_n_pop = model.add_neuron_population(
        "ManualPostDenseNeuron", 4, post_neuron_model,
        {}, {"x": 0.0})
    model.add_synapse_population(
        "ManualPostDenseSynapse", "DENSE", 0,
        ss_pop, manual_dense_n_pop,
        "StaticPulse", {}, {"g": dense.flatten()}, {}, {},
        "DeltaCurr", {}, {})

    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    output_place_values = 2 ** np.arange(4)
    output_populations = [sparse_constant_weight_n_pop,
                          sparse_constant_weight_pre_n_pop,
                          manual_sparse_constant_weight_n_pop,
                          sparse_n_pop, sparse_pre_n_pop, manual_sparse_n_pop,
                          bitmask_n_pop, dense_n_pop, manual_dense_n_pop]
    while model.timestep < 16:
        model.step_time()

        # Loop through output populations
        for pop in output_populations:
            # Pull state variable
            pop.pull_var_from_device("x")

            # Convert to binary mask
            output_binary = np.isclose(np.ones(4), pop.vars["x"].view)

            # Sum up active place values
            output_value = np.sum(output_place_values[output_binary])
            if output_value != (model.timestep - 1):
                assert False, f"{pop.name} decoding incorrect ({output_value} rather than {model.timestep - 1})"

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_forward_den_delay(backend, precision):
    model = GeNNModel(precision, "test_forward_den_delay", backend=backend)
    model.dt = 1.0

    # Create spike source array to generate one-hot pattern to decode
    ss_pop = model.add_neuron_population("SpikeSource", 10, "SpikeSourceArray",
                                         {}, {"startSpike": np.arange(10), "endSpike": np.arange(1, 11)})
    ss_pop.extra_global_params["spikeTimes"].set_values(np.arange(10.0))

    # Create one output neuron pop with dense decoder population
    delay = np.arange(9, -1, -1)
    dense_n_pop = model.add_neuron_population(
        "PostDenseNeuron", 1, post_neuron_model,
        {}, {"x": 0.0})
    dense_s_pop = model.add_synapse_population(
        "PostDenseSynapse", "DENSE", 0,
        ss_pop, dense_n_pop,
        "StaticPulseDendriticDelay", {}, {"g": 1.0, "d": delay}, {}, {},
        "DeltaCurr", {}, {})
    dense_s_pop.max_dendritic_delay_timesteps = 10

    # Create one output neuron pop with sparse decoder population
    sparse_n_pop = model.add_neuron_population(
        "PostSparseNeuron", 1, post_neuron_model,
        {}, {"x": 0.0})
    sparse_s_pop = model.add_synapse_population(
        "PostSparseSynapse", "SPARSE", 0,
        ss_pop, sparse_n_pop,
        "StaticPulseDendriticDelay", {}, {"g": 1.0, "d": delay}, {}, {},
        "DeltaCurr", {}, {})
    sparse_s_pop.max_dendritic_delay_timesteps = 10
    sparse_s_pop.set_sparse_connections(np.arange(10), np.zeros(10, dtype=int))
    
    # Create one output neuron pop with sparse decoder population and presynaptic parallelism
    sparse_pre_n_pop = model.add_neuron_population(
        "PostSparsePreSpanNeuron", 1, post_neuron_model,
        {}, {"x": 0.0})
    sparse_pre_s_pop = model.add_synapse_population(
        "PostSparsePreSpanSynapse", "SPARSE", 0,
        ss_pop, sparse_pre_n_pop,
        "StaticPulseDendriticDelay", {}, {"g": 1.0, "d": delay}, {}, {},
        "DeltaCurr", {}, {})
    sparse_pre_s_pop.max_dendritic_delay_timesteps = 10
    sparse_pre_s_pop.set_sparse_connections(np.arange(10), np.zeros(10, dtype=int))
    sparse_pre_s_pop.span_type = SpanType.PRESYNAPTIC

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
            pop.pull_var_from_device("x")

            # If not close to correct value, error
            if not np.isclose(pop.vars["x"].view[0], correct):
                assert False, f"{pop.name} decoding incorrect ({pop.vars['x'].view[0]} rather than {correct})"

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_reverse(backend, precision):
    model = GeNNModel(precision, "test_reverse", backend=backend)
    model.dt = 1.0

    # Create spike source array with extra x variable 
    # to generate one-hot pattern to decode
    pre_n_pop = model.add_neuron_population(
        "SpikeSource", 16, pre_reverse_spike_source_model,
        {}, {"startSpike": np.arange(16), "endSpike": np.arange(1, 17), "x": 0.0})
    pre_n_pop.extra_global_params["spikeTimes"].set_values(np.arange(16.0))
    
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
        "SparseSynapse", "SPARSE", 0,
        pre_n_pop, post_n_pop,
        static_pulse_reverse_model, {}, {"g": weights}, {}, {},
        "DeltaCurr", {}, {})
    s_pop.set_sparse_connections(pre_inds, post_inds)
    
    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    while model.timestep < 16:
        model.step_time()
        
        pre_n_pop.pull_var_from_device("x")
        assert np.sum(pre_n_pop.vars["x"].view) == (model.timestep - 1)

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_reverse_post(backend, precision):
    model = GeNNModel(precision, "test_reverse_post", backend=backend)
    model.dt = 1.0

    # Add presynaptic populations to sum reverse input
    sparse_pre_n_pop = model.add_neuron_population(
        "SparsePost", 4, pre_reverse_model,
        {}, {"x": 0.0})
    dense_pre_n_pop = model.add_neuron_population(
        "DensePost", 4, pre_reverse_model,
        {}, {"x": 0.0})
        
    # Create spike source array to generate one-hot pattern to decode
    post_n_pop = model.add_neuron_population(
        "SpikeSource", 16, "SpikeSourceArray",
        {}, {"startSpike": np.arange(16), "endSpike": np.arange(1, 17)})
    post_n_pop.extra_global_params["spikeTimes"].set_values(np.arange(16.0))

    # Build sparse connectivity
    pre_inds = []
    post_inds = []
    for j in range(16):
        for i in range(4):
            i_value = 1 << i
            if ((j + 1) & i_value) != 0:
                pre_inds.append(i)
                post_inds.append(j)

    pre_inds = np.asarray(pre_inds)
    post_inds = np.asarray(post_inds)

    # Use to build dense matrix
    dense = np.zeros((4, 16))
    dense[pre_inds,post_inds] = 1.0

    # Add connectivity
    sparse_s_pop = model.add_synapse_population(
        "SparseSynapse", "SPARSE", 0,
        sparse_pre_n_pop, post_n_pop,
        static_pulse_reverse_post_model, {}, {"g": 1.0}, {}, {},
        "DeltaCurr", {}, {})
    sparse_s_pop.set_sparse_connections(pre_inds, post_inds)
    model.add_synapse_population(
        "DenseSynapse", "DENSE", 0,
        dense_pre_n_pop, post_n_pop,
        static_pulse_reverse_post_model, {}, {"g": dense.flatten()}, {}, {},
        "DeltaCurr", {}, {})
        
    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    output_place_values = 2 ** np.arange(4)
    output_populations = [sparse_pre_n_pop, dense_pre_n_pop]
    while model.timestep < 16:
        model.step_time()
        
        # Loop through output populations
        for pop in output_populations:
            # Pull state variable
            pop.pull_var_from_device("x")

            # Convert to binary mask
            output_binary = np.isclose(np.ones(4), pop.vars["x"].view)

            # Sum up active place values
            output_value = np.sum(output_place_values[output_binary])
            if output_value != (model.timestep - 1):
                assert False, f"{pop.name} decoding incorrect ({output_value} rather than {model.timestep - 1})"

if __name__ == '__main__':
    test_forward_den_delay("cuda", types.Float)
    