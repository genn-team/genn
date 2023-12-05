import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel

from pygenn.genn import SpanType, VarAccess
from pygenn import (create_neuron_model,
                    create_sparse_connect_init_snippet,
                    create_var_init_snippet,
                    create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity, 
                    init_toeplitz_connectivity,
                    init_weight_update, init_var)

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

static_event_pulse_model = create_weight_update_model(
    "static_event_pulse",
    var_name_types=[("g", "scalar")],
    event_threshold_condition_code=
    """
    (unsigned int)round(t) == id
    """,
    event_code=
    """
    addToPost(g);
    """)

# (Normalised) horizontal Sobel convolution kernel
vertical_kernel = np.asarray([[1.0,   0.0,    -1.0],
                              [2.0,   0.0,    -2.0],
                              [1.0,   0.0,    -1.0]])

# (Normalised) vertical Sobel convolution kernel
horizontal_kernel = np.asarray([[1.0,     2.0,    1.0],
                                [0.0,     0.0,    0.0],
                                [-1.0,    -2.0,   -1.0]])

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_forward(backend, precision):    
    model = GeNNModel(precision, "test_forward", backend=backend)
    model.dt = 1.0

    # Create spike source array to generate one-hot pattern to decode
    ss_pop = model.add_neuron_population("SpikeSource", 16, "SpikeSourceArray",
                                         {}, {"startSpike": np.arange(16), "endSpike": np.arange(1, 17)})
    ss_pop.extra_global_params["spikeTimes"].set_init_values(np.arange(16.0))

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
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))
    
    # Create one output neuron pop with constant weight 
    # sparse decoder population and presynaptic parallelism
    sparse_constant_weight_pre_n_pop = model.add_neuron_population(
        "PostSparseConstantWeightPreSpanNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    sparse_constant_weight_pre_s_pop = model.add_synapse_population(
        "SparseConstantWeightPreSpanSynapse", "SPARSE", 0,
        ss_pop, sparse_constant_weight_pre_n_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))
    sparse_constant_weight_pre_s_pop.span_type = SpanType.PRESYNAPTIC

    # Create one output neuron pop with constant weight sparse decoder population
    manual_sparse_constant_weight_n_pop = model.add_neuron_population(
        "ManualPostSparseConstantWeightNeuron", 4, post_neuron_model,
        {}, {"x": 0.0})
    manual_sparse_constant_weight_s_pop = model.add_synapse_population(
        "ManualSparseConstantWeightSynapse", "SPARSE", 0,
        ss_pop, manual_sparse_constant_weight_n_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"))
    manual_sparse_constant_weight_s_pop.set_sparse_connections(pre_inds,
                                                               post_inds)

    # Create one output neuron pop with sparse decoder population
    sparse_n_pop = model.add_neuron_population(
        "PostSparseNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "SparseSynapse", "SPARSE", 0,
        ss_pop, sparse_n_pop,
        init_weight_update("StaticPulse", {}, {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))
    
    # Create one output neuron pop with sparse 
    # decoder population and presynaptic parallelism
    sparse_pre_n_pop = model.add_neuron_population(
        "PostSparsePreSpanNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    sparse_pre_s_pop = model.add_synapse_population(
        "SparsePreSpanSynapse", "SPARSE", 0,
        ss_pop, sparse_pre_n_pop,
        init_weight_update("StaticPulse", {}, {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))
    sparse_pre_s_pop.span_type = SpanType.PRESYNAPTIC
    
    # Create one output neuron pop with sparse 
    # decoder population and presynaptic parallelism
    sparse_hybrid_n_pop = model.add_neuron_population(
        "PostSparseHybridNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    sparse_hybrid_s_pop = model.add_synapse_population(
        "SparseHybridSynapse", "SPARSE", 0,
        ss_pop, sparse_hybrid_n_pop,
 
        init_weight_update("StaticPulse", {}, {"g": 1.0}, {}, {}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))
    sparse_hybrid_s_pop.span_type = SpanType.PRESYNAPTIC
    sparse_hybrid_s_pop.num_threads_per_spike = 2

    # Create one output neuron pop with sparse decoder population
    manual_sparse_n_pop = model.add_neuron_population(
        "ManualPostSparseNeuron", 4, post_neuron_model,
        {}, {"x": 0.0})
    manual_sparse_s_pop = model.add_synapse_population(
        "ManualSparseSynapse", "SPARSE", 0,
        ss_pop, manual_sparse_n_pop,
        init_weight_update("StaticPulse", {}, {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))
    manual_sparse_s_pop.set_sparse_connections(pre_inds, post_inds)

    # Create one output neuron pop with bitmask decoder population
    bitmask_n_pop = model.add_neuron_population(
        "PostBitmaskNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "BitmaskSynapse", "SPARSE", 0,
        ss_pop, bitmask_n_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))

    # Create one output neuron pop with dense decoder population
    dense_n_pop = model.add_neuron_population(
        "PostDenseNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "PostDenseSynapse", "DENSE", 0,
        ss_pop, dense_n_pop,
        init_weight_update("StaticPulse", {}, {"g": init_var(decoder_dense_model, {})}),
        init_postsynaptic("DeltaCurr"))

    # Create one output neuron pop with dense decoder population
    manual_dense_n_pop = model.add_neuron_population(
        "ManualPostDenseNeuron", 4, post_neuron_model,
        {}, {"x": 0.0})
    model.add_synapse_population(
        "ManualPostDenseSynapse", "DENSE", 0,
        ss_pop, manual_dense_n_pop,
        init_weight_update("StaticPulse", {}, {"g": dense.flatten()}),
        init_postsynaptic("DeltaCurr", {}, {}))

    # Create one output neuron pop with sparse decoder population driven by spike-like evnets
    sparse_event_n_pop = model.add_neuron_population(
        "PostSparseEventNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "SparseEventSynapse", "SPARSE", 0,
        ss_pop, sparse_event_n_pop,
        init_weight_update(static_event_pulse_model, {}, {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model, {}))

    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    output_place_values = 2 ** np.arange(4)
    output_populations = [sparse_constant_weight_n_pop,
                          sparse_constant_weight_pre_n_pop,
                          manual_sparse_constant_weight_n_pop,
                          sparse_n_pop, sparse_pre_n_pop, sparse_hybrid_n_pop,
                          manual_sparse_n_pop, bitmask_n_pop, dense_n_pop, 
                          manual_dense_n_pop, sparse_event_n_pop]
    while model.timestep < 16:
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
def test_forward_den_delay(backend, precision):
    model = GeNNModel(precision, "test_forward_den_delay", backend=backend)
    model.dt = 1.0

    # Create spike source array to generate one-hot pattern to decode
    ss_pop = model.add_neuron_population("SpikeSource", 10, "SpikeSourceArray",
                                         {}, {"startSpike": np.arange(10), "endSpike": np.arange(1, 11)})
    ss_pop.extra_global_params["spikeTimes"].set_init_values(np.arange(10.0))

    # Create one output neuron pop with dense decoder population
    delay = np.arange(9, -1, -1)
    dense_n_pop = model.add_neuron_population(
        "PostDenseNeuron", 1, post_neuron_model,
        {}, {"x": 0.0})
    dense_s_pop = model.add_synapse_population(
        "PostDenseSynapse", "DENSE", 0,
        ss_pop, dense_n_pop,
        init_weight_update("StaticPulseDendriticDelay", {}, {"g": 1.0, "d": delay}),
        init_postsynaptic("DeltaCurr", {}, {}))
    dense_s_pop.max_dendritic_delay_timesteps = 10

    # Create one output neuron pop with sparse decoder population
    sparse_n_pop = model.add_neuron_population(
        "PostSparseNeuron", 1, post_neuron_model,
        {}, {"x": 0.0})
    sparse_s_pop = model.add_synapse_population(
        "PostSparseSynapse", "SPARSE", 0,
        ss_pop, sparse_n_pop,
        init_weight_update("StaticPulseDendriticDelay", {}, {"g": 1.0, "d": delay}),
        init_postsynaptic("DeltaCurr", {}, {}))
    sparse_s_pop.max_dendritic_delay_timesteps = 10
    sparse_s_pop.set_sparse_connections(np.arange(10), np.zeros(10, dtype=int))
    
    # Create one output neuron pop with sparse decoder population and presynaptic parallelism
    sparse_pre_n_pop = model.add_neuron_population(
        "PostSparsePreSpanNeuron", 1, post_neuron_model,
        {}, {"x": 0.0})
    sparse_pre_s_pop = model.add_synapse_population(
        "PostSparsePreSpanSynapse", "SPARSE", 0,
        ss_pop, sparse_pre_n_pop, 
        init_weight_update("StaticPulseDendriticDelay", {}, {"g": 1.0, "d": delay}),
        init_postsynaptic("DeltaCurr", {}, {}))
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
            pop.vars["x"].pull_from_device()

            # If not close to correct value, error
            if not np.isclose(pop.vars["x"].view[0], correct):
                assert False, f"{pop.name} decoding incorrect ({pop.vars['x'].view[0]} rather than {correct})"


@pytest.mark.parametrize("backend", ["cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_forward_procedural(backend, precision):
    model = GeNNModel(precision, "test_forward_procedural", backend=backend)
    model.dt = 1.0

    # Create spike source array to generate one-hot pattern to decode
    ss_pop = model.add_neuron_population("SpikeSource", 16, "SpikeSourceArray",
                                         {}, {"startSpike": np.arange(16), "endSpike": np.arange(1, 17)})
    ss_pop.extra_global_params["spikeTimes"].set_init_values(np.arange(16.0))

    # Create one output neuron pop with constant weight procedural decoder population
    procedural_constant_weight_n_pop = model.add_neuron_population(
        "PostProceduralConstantWeightNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "ProceduralConstantWeightSynapse", "PROCEDURAL", 0,
        ss_pop, procedural_constant_weight_n_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model))
    
    # Create one output neuron pop with dense procedural decoder population
    dense_procedural_n_pop = model.add_neuron_population(
        "PostDenseProceduralNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "DenseProceduralSynapse", "DENSE_PROCEDURALG", 0,
        ss_pop, dense_procedural_n_pop,
        init_weight_update("StaticPulse", {}, {"g": init_var(decoder_dense_model, {})}),
        init_postsynaptic("DeltaCurr"))
    
    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    output_place_values = 2 ** np.arange(4)
    output_populations = [procedural_constant_weight_n_pop,
                          dense_procedural_n_pop]
    while model.timestep < 16:
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
def test_forward_kernel(backend, precision):
    model = GeNNModel(precision, "test_forward_kernel", backend=backend)
    model.dt = 1.0

    # Create spike source array to present test pattern
    test_pattern = np.load("test_pattern.npy")
    end_spikes = np.cumsum(np.bincount(test_pattern, minlength=64 * 64))
    start_spikes = np.concatenate(([0,], end_spikes[:-1]))
    pre_pop = model.add_neuron_population("SpikeSource", 64 * 64, "SpikeSourceArray",
                                          {}, {"startSpike": start_spikes, "endSpike": end_spikes})
    pre_pop.extra_global_params["spikeTimes"].set_init_values(np.zeros_like(test_pattern))

    # Add postsynaptic populations to receive horizontal and vertical edges
    post_toeplitz_horiz_pop = model.add_neuron_population(
        "PostHorizNeurons", 62 * 62, post_neuron_model, 
        {}, {"x": 0.0})

    post_toeplitz_vert_pop = model.add_neuron_population(
        "PostVertNeurons", 62 * 62, post_neuron_model, 
        {}, {"x": 0.0})

    post_sparse_horiz_pop = model.add_neuron_population(
        "PostSparseHorizNeurons", 62 * 62, post_neuron_model, 
        {}, {"x": 0.0})

    post_sparse_vert_pop = model.add_neuron_population(
        "PostSparseVertNeurons", 62 * 62, post_neuron_model, 
        {}, {"x": 0.0})
    
    # Add convolutional toeplitz connectivity
    conv_toeplitz_params = {"conv_kh": 3, "conv_kw": 3,
                            "conv_ih": 64, "conv_iw": 64, "conv_ic": 1,
                            "conv_oh": 62, "conv_ow": 62, "conv_oc": 1}
    model.add_synapse_population(
        "ToeplitzHorizSynapse", "TOEPLITZ", 0,
        pre_pop, post_toeplitz_horiz_pop,
 
        init_weight_update("StaticPulse", {}, {"g": horizontal_kernel.flatten()}),
        init_postsynaptic("DeltaCurr"),
        init_toeplitz_connectivity("Conv2D", conv_toeplitz_params))
    model.add_synapse_population(
        "ToeplitzVertSynapse", "TOEPLITZ", 0,
        pre_pop, post_toeplitz_vert_pop,
 
        init_weight_update("StaticPulse", {}, {"g": vertical_kernel.flatten()}),
        init_postsynaptic("DeltaCurr"),
        init_toeplitz_connectivity("Conv2D", conv_toeplitz_params))

    # Add sparse connectivity with kernel initialisation
    conv_params = {"conv_kh": 3, "conv_kw": 3,
                   "conv_sh": 1, "conv_sw": 1,
                   "conv_padh": 0, "conv_padw": 0,
                   "conv_ih": 64, "conv_iw": 64, "conv_ic": 1,
                   "conv_oh": 62, "conv_ow": 62, "conv_oc": 1}
    sparse_horiz_s_pop = model.add_synapse_population(
        "SparseHorizSynapse", "SPARSE", 0,
        pre_pop, post_sparse_horiz_pop,
 
        init_weight_update("StaticPulse", {}, {"g": init_var("Kernel")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("Conv2D", conv_params))
    sparse_horiz_s_pop.vars["g"].extra_global_params["kernel"].set_init_values(horizontal_kernel.flatten())

    sparse_vert_s_pop = model.add_synapse_population(
        "SparseVertSynapse", "SPARSE", 0,
        pre_pop, post_sparse_vert_pop,
 
        init_weight_update("StaticPulse", {}, {"g": init_var("Kernel")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("Conv2D", conv_params))
    sparse_vert_s_pop.vars["g"].extra_global_params["kernel"].set_init_values(vertical_kernel.flatten())

    # Build model and load
    model.build()
    model.load()
    
    # Step time twice - in first timestep spikes will be emitted 
    # by pre_pop. In second, they will be received by the post_pops
    model.step_time()
    model.step_time()
    
    # Download output variables from device
    post_toeplitz_horiz_pop.vars["x"].pull_from_device()
    post_toeplitz_vert_pop.vars["x"].pull_from_device()
    post_sparse_horiz_pop.vars["x"].pull_from_device()
    post_sparse_vert_pop.vars["x"].pull_from_device()

    # Check against correct convolutions
    correct_horizontal = np.load("horizontal_output.npy") 
    correct_vertical = np.load("vertical_output.npy")
    assert np.allclose(post_toeplitz_horiz_pop.vars["x"].view, 
                       correct_horizontal)
    assert np.allclose(post_toeplitz_vert_pop.vars["x"].view, 
                       correct_vertical)
    assert np.allclose(post_sparse_horiz_pop.vars["x"].view, 
                       correct_horizontal)
    assert np.allclose(post_sparse_vert_pop.vars["x"].view, 
                       correct_vertical)

@pytest.mark.parametrize("backend", ["cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_forward_kernel_procedural(backend, precision):
    model = GeNNModel(precision, "test_forward_kernel_procedural", backend=backend)
    model.dt = 1.0

    # Create spike source array to present test pattern
    test_pattern = np.load("test_pattern.npy")
    end_spikes = np.cumsum(np.bincount(test_pattern, minlength=64 * 64))
    start_spikes = np.concatenate(([0,], end_spikes[:-1]))
    pre_pop = model.add_neuron_population("SpikeSource", 64 * 64, "SpikeSourceArray",
                                          {}, {"startSpike": start_spikes, "endSpike": end_spikes})
    pre_pop.extra_global_params["spikeTimes"].set_init_values(np.zeros_like(test_pattern))

    # Add two postsynaptic populations to receive horizontal and vertical edges
    post_horiz_pop = model.add_neuron_population(
        "PostHorizNeurons", 62 * 62, post_neuron_model, 
        {}, {"x": 0.0})

    post_vert_pop = model.add_neuron_population(
        "PostVertNeurons", 62 * 62, post_neuron_model, 
        {}, {"x": 0.0})

    # Add convolutional toeplitz connectivity
    conv_params = {"conv_kh": 3, "conv_kw": 3,
                   "conv_sh": 1, "conv_sw": 1,
                   "conv_padh": 0, "conv_padw": 0,
                   "conv_ih": 64, "conv_iw": 64, "conv_ic": 1,
                   "conv_oh": 62, "conv_ow": 62, "conv_oc": 1}
    model.add_synapse_population(
        "HorizSynapse", "PROCEDURAL_KERNELG", 0,
        pre_pop, post_horiz_pop,
        init_weight_update("StaticPulse", {}, {"g": horizontal_kernel.flatten()}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("Conv2D", conv_params))
    model.add_synapse_population(
        "VertSynapse", "PROCEDURAL_KERNELG", 0,
        pre_pop, post_vert_pop,
        init_weight_update("StaticPulse", {}, {"g": vertical_kernel.flatten()}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("Conv2D", conv_params))

    # Build model and load
    model.build()
    model.load()
    
    # Step time twice - in first timestep spikes will be emitted 
    # by pre_pop. In second, they will be received by the post_pops
    model.step_time()
    model.step_time()
    
    # Download output variables from device
    post_horiz_pop.vars["x"].pull_from_device()
    post_vert_pop.vars["x"].pull_from_device()
    
    # Check against correct convolutions
    assert np.allclose(post_horiz_pop.vars["x"].view, 
                       np.load("horizontal_output.npy"))
    assert np.allclose(post_vert_pop.vars["x"].view, 
                       np.load("vertical_output.npy"))

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_reverse(backend, precision):
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
    
    pre_model = create_neuron_model(
        "pre",
        var_name_types=[("x", "scalar")],
        extra_global_params=[("spikeTimes", "scalar*")],
        sim_code=
        """
        x = Isyn;
        """)

    static_pulse_reverse_model = create_weight_update_model(
        "static_pulse_reverse",
        sim_code=
        """
        $(addToPre, $(g));
        """,
        var_name_types=[("g", "scalar", VarAccess.READ_ONLY)])


    static_event_pulse_reverse_model = create_weight_update_model(
        "static_event_pulse_reverse",
        var_name_types=[("g", "scalar")],
        event_threshold_condition_code=
        """
        (unsigned int)round(t) == id
        """,
        event_code=
        """
        addToPre(g);
        """)


    model = GeNNModel(precision, "test_reverse", backend=backend)
    model.dt = 1.0

    # Create spike source arrays with extra x variable 
    # to generate one-hot pattern to decode
    pre_n_pop = model.add_neuron_population(
        "SpikeSource", 16, pre_reverse_spike_source_model,
        {}, {"startSpike": np.arange(16), "endSpike": np.arange(1, 17), "x": 0.0})
    pre_n_pop.extra_global_params["spikeTimes"].set_init_values(np.arange(16.0))
    pre_pre_n_pop = model.add_neuron_population(
        "PreSpikeSource", 16, pre_reverse_spike_source_model,
        {}, {"startSpike": np.arange(16), "endSpike": np.arange(1, 17), "x": 0.0})
    pre_pre_n_pop.extra_global_params["spikeTimes"].set_init_values(np.arange(16.0))
    pre_event_n_pop = model.add_neuron_population(
        "EventSpikeSource", 16, pre_model,
        {}, {"x": 0.0})

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
        init_weight_update(static_pulse_reverse_model, {}, {"g": weights}),
        init_postsynaptic("DeltaCurr"))
    s_pop.set_sparse_connections(pre_inds, post_inds)
    
    s_pre_pop = model.add_synapse_population(
        "SparsePreSynapse", "SPARSE", 0,
        pre_pre_n_pop, post_n_pop,
        init_weight_update(static_pulse_reverse_model, {}, {"g": weights}),
        init_postsynaptic("DeltaCurr"))
    s_pre_pop.set_sparse_connections(pre_inds, post_inds)
    s_pre_pop.span_type = SpanType.PRESYNAPTIC

    s_event_pop = model.add_synapse_population(
        "SparseEventSynapse", "SPARSE", 0,
        pre_event_n_pop, post_n_pop,
        init_weight_update(static_event_pulse_reverse_model, {}, {"g": weights}),
        init_postsynaptic("DeltaCurr"))
    s_event_pop.set_sparse_connections(pre_inds, post_inds)
    
    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    while model.timestep < 16:
        model.step_time()
        
        pre_n_pop.vars["x"].pull_from_device()
        pre_pre_n_pop.vars["x"].pull_from_device()
        pre_event_n_pop.vars["x"].pull_from_device()

        assert np.sum(pre_n_pop.vars["x"].view) == (model.timestep - 1)
        assert np.sum(pre_pre_n_pop.vars["x"].view) == (model.timestep - 1)
        assert np.sum(pre_event_n_pop.vars["x"].view) == (model.timestep - 1)

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_reverse_post(backend, precision):
    pre_reverse_model = create_neuron_model(
        "pre_reverse",
        var_name_types=[("x", "scalar")],
        sim_code=
        """
        x = Isyn;
        """)

    static_pulse_reverse_post_model = create_weight_update_model(
        "static_pulse_reverse_post",
        learn_post_code=
        """
        $(addToPre, $(g));
        """,
        var_name_types=[("g", "scalar", VarAccess.READ_ONLY)])
    
    static_pulse_reverse_event_post_model = create_weight_update_model(
        "static_pulse_reverse_event_post",
        post_event_code=
        """
        $(addToPre, $(g));
        """,
        post_event_threshold_condition_code=
        """
        (unsigned int)round(t) == id
        """,
        var_name_types=[("g", "scalar", VarAccess.READ_ONLY)])

    model = GeNNModel(precision, "test_reverse_post", backend=backend)
    model.dt = 1.0

    # Add presynaptic populations to sum reverse input
    sparse_pre_n_pop = model.add_neuron_population(
        "SparsePost", 4, pre_reverse_model,
        {}, {"x": 0.0})
    dense_pre_n_pop = model.add_neuron_population(
        "DensePost", 4, pre_reverse_model,
        {}, {"x": 0.0})
    sparse_event_pre_n_pop = model.add_neuron_population(
        "SparseEvemtPost", 4, pre_reverse_model,
        {}, {"x": 0.0})
        
    # Create spike source array to generate one-hot pattern to decode
    post_n_pop = model.add_neuron_population(
        "SpikeSource", 16, "SpikeSourceArray",
        {}, {"startSpike": np.arange(16), "endSpike": np.arange(1, 17)})
    post_n_pop.extra_global_params["spikeTimes"].set_init_values(np.arange(16.0))

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
        init_weight_update(static_pulse_reverse_post_model, {}, {"g": 1.0}),
        init_postsynaptic("DeltaCurr"))
    sparse_s_pop.set_sparse_connections(pre_inds, post_inds)
    model.add_synapse_population(
        "DenseSynapse", "DENSE", 0,
        dense_pre_n_pop, post_n_pop,
        init_weight_update(static_pulse_reverse_post_model, {}, {"g": dense.flatten()}),
        init_postsynaptic("DeltaCurr"))
    sparse_event_s_pop = model.add_synapse_population(
        "SparseEventSynapse", "SPARSE", 0,
        sparse_event_pre_n_pop, post_n_pop,
        init_weight_update(static_pulse_reverse_event_post_model, {}, {"g": 1.0}),
        init_postsynaptic("DeltaCurr"))
    sparse_event_s_pop.set_sparse_connections(pre_inds, post_inds)

    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    output_place_values = 2 ** np.arange(4)
    output_populations = [sparse_pre_n_pop, sparse_event_pre_n_pop, dense_pre_n_pop]
    while model.timestep < 16:
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

if __name__ == '__main__':
    test_forward("cuda", types.Float)
    
