import numpy as np
import pytest
from pygenn import types

from pygenn import VarAccessMode, VarAccess

from bitarray import bitarray
from bitarray.util import hex2ba
from pygenn import (create_custom_connectivity_update_model,
                    create_neuron_model,
                    create_weight_update_model,
                    create_sparse_connect_init_snippet,
                    create_var_init_snippet,
                    create_var_ref,
                    create_wu_var_ref,
                    create_wu_pre_var_ref,
                    create_wu_post_var_ref,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_var, init_weight_update)

# Neuron model which does nothing
empty_neuron_model = create_neuron_model("empty")

# Snippet to initialise variable to hold its column-major index
weight_init_snippet = create_var_init_snippet(
    "weight_init",
    var_init_code=
    """
    value = (id_pre * 64) + id_post;
    """)

def _clear_bit(ba, bit):
    ba_copy = ba.copy()
    ba_copy[bit] = False
    return ba_copy

def _check_connectivity(sg, get_row_length_fn, get_connectivity_fn, var_checks=[]):
    sg.pull_connectivity_from_device()

    # Pull all variables from device and get numpy array of values
    var_values = []
    for pop, var_name, _ in var_checks:
        pop.vars[var_name].pull_from_device()
        var_values.append(pop.vars[var_name].values)

    pre_inds = sg.get_sparse_pre_inds()
    post_inds = sg.get_sparse_post_inds()

    # Loop through rows
    row_lengths = np.bincount(pre_inds, minlength=sg.src.num_neurons)
    for i in range(sg.src.num_neurons):
        # Check row lengths
        assert row_lengths[i] == get_row_length_fn(i)

        # Build mask of row
        row_mask = (pre_inds == i)
        row_inds = post_inds[row_mask]

        # Build bitarray of row connectivity
        # **YUCK** converting to list
        row_bits = bitarray(sg.trg.num_neurons)
        row_bits.setall(0)
        row_bits[list(row_inds)] = 1
        
        # Check connectivity
        assert row_bits == get_connectivity_fn(i)

        # Loop through variables and check values match chosen pattern
        for (pop, var_name, transpose), val in zip(var_checks, var_values):
            correct = ((row_inds * 64.0) + i if transpose 
                       else (i * 64.0) + row_inds)
            if len(val.shape) == 1:
                assert np.allclose(val[row_mask], correct)
            else:
                assert np.allclose(val[:,row_mask], correct)



@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_connectivity_update(make_model, backend, precision, batch_size):
    neuron_model = create_neuron_model(
        "neuron",
        vars=[("V", "scalar")])

    weight_update_model = create_weight_update_model(
        "weight_update",
        vars=[("g", "scalar", VarAccess.READ_ONLY_DUPLICATE),
                        ("d", "unsigned int", VarAccess.READ_ONLY)])

    # Snippet to initialise variable to hold its row-major index
    delay_init_snippet = create_var_init_snippet(
        "delay_init",
        var_init_code=
        """
        value = (id_post * 64) + id_pre;
        """)

    # Snippet to configure 'triangle' sparse connectivity where presynaptic neurons
    # are connected to all postsynaptic neurons with the same or higher ID
    triangle_connect_init_snippet = create_sparse_connect_init_snippet(
        "triangle_connect_init",
        row_build_code=
        """
        for(unsigned int j = id_pre; j < num_post; j++) {
            addSynapse(j);
        }
        """)

    # Custom connectivity update which removes synapses on diagonal
    remove_synapse_model = create_custom_connectivity_update_model(
        "remove_synapse",
        vars=[("a", "scalar")],
        row_update_code=
        """
        for_each_synapse {
            if(id_post == id_pre) {
                remove_synapse();
                break;
            }
        }
        """)

    # Custom connectivity update which removes synapses 
    # based on bitset EGP configured in host code
    remove_synapse_host_egp_model = create_custom_connectivity_update_model(
        "remove_synapse_host_egp",
        extra_global_params=[("d", "uint32_t*")],
        row_update_code=
        """
        const unsigned int wordsPerRow = (num_post + 31) / 32;
        for_each_synapse {
            if(d[(wordsPerRow * id_pre) + (id_post / 32)] & (1 << (id_post % 32))) {
                remove_synapse();
                break;
            }
        }
        """,
        host_update_code=
        """
        const unsigned int wordsPerRow = (num_post + 31) / 32;
        for(unsigned int i = 0; i < wordsPerRow * num_pre; i++) {
            d[i] = 0;
        }
        
        for(unsigned int i = 0; i < num_pre; i++) {
            uint32_t *dRow = d + (wordsPerRow * i);
            dRow[i / 32] |= (1 << (i % 32));
        }
        pushdToDevice(wordsPerRow * num_pre);
        """)

    # Custom connectivity update which removes one synapse per row 
    # based on presynaptic variable set in host code
    remove_synapse_host_pre_var_model = create_custom_connectivity_update_model(
        "remove_synapse_host_pre_var",
        pre_vars=[("postInd", "unsigned int")],
        row_update_code=
        """
        for_each_synapse {
            if(id_post == postInd) {
                remove_synapse();
                break;
            }
        }
        """,
        host_update_code=
        """
        for(unsigned int i = 0; i < num_pre; i++) {
            postInd[i] = i;
        }
        pushpostIndToDevice();
        """)

    # Custom connectivity update which adds a synapse on diagonal to each row
    add_synapse_model = create_custom_connectivity_update_model(
        "add_synapse",
        var_refs=[("g", "scalar"), ("d", "unsigned int"), ("a", "scalar")],
        row_update_code=
        """
        const scalar weight = (id_pre * 64) + id_pre;
        const unsigned int delay = (id_pre * 64) + id_pre;
        add_synapse(id_pre, weight, delay, weight);
        """)

    model = make_model(precision, "test_custom_connectivity_update", backend=backend)
    model.batch_size = batch_size
    model.dt = 1.0

    # Create pre and postsynaptic populations
    pre_n_pop = model.add_neuron_population("PreNeurons", 64, empty_neuron_model); 
    post_n_pop = model.add_neuron_population("PostNeurons", 64, empty_neuron_model); 

    # Create synapse groups
    s_pop_1 = model.add_synapse_population(
        "Syn1", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(weight_update_model, {}, {"g": init_var(weight_init_snippet), "d": init_var(delay_init_snippet)}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(triangle_connect_init_snippet))

    s_pop_2 = model.add_synapse_population(
        "Syn2", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(weight_update_model, {}, {"g": init_var(weight_init_snippet), "d": init_var(delay_init_snippet)}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(triangle_connect_init_snippet))
    
    s_pop_3 = model.add_synapse_population(
        "Syn3", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(weight_update_model, {}, {"g": init_var(weight_init_snippet), "d": init_var(delay_init_snippet)}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(triangle_connect_init_snippet))

    # Create custom connectivity updates
    remove_synapse_ccu = model.add_custom_connectivity_update(
        "RemoveSynapse", "RemoveSynapse", s_pop_1,
        remove_synapse_model,
        {}, {"a": init_var(weight_init_snippet)}, {}, {}, 
        {}, {}, {})
    
    remove_synapse_host_egp_ccu = model.add_custom_connectivity_update(
        "RemoveSynapseHostEGP", "RemoveSynapse", s_pop_2,
        remove_synapse_host_egp_model,
        {}, {}, {}, {}, 
        {}, {}, {})
    num_words = post_n_pop.num_neurons * ((pre_n_pop.num_neurons + 31) // 32)
    remove_synapse_host_egp_ccu.extra_global_params["d"].set_init_values(
        np.empty(num_words, dtype=np.uint32))
    
    remove_synapse_host_pre_var_ccu = model.add_custom_connectivity_update(
        "RemoveSynapseHostPreVar", "RemoveSynapse", s_pop_3,
        remove_synapse_host_pre_var_model,
        {}, {}, {"postInd": 0}, {}, 
        {}, {}, {})

    add_synapse_ccu = model.add_custom_connectivity_update(
        "AddSynapse", "AddSynapse", s_pop_1,
        add_synapse_model,
        {}, {}, {}, {},
        {"g": create_wu_var_ref(s_pop_1, "g"), "d": create_wu_var_ref(s_pop_1, "d"), "a": create_wu_var_ref(remove_synapse_ccu, "a")}, {}, {})

    # Build model and load
    model.build()
    model.load()
    
    samples = {s_pop_1: [(s_pop_1, "g", False), (s_pop_1, "d", True), (remove_synapse_ccu, "a", False)],
               s_pop_2: [(s_pop_2, "g", False), (s_pop_2, "d", True)],
               s_pop_3: [(s_pop_3, "g", False), (s_pop_3, "d", True)]}
                
    # Check initial connectivity
    for pop, var_checks in samples.items():
        _check_connectivity(pop, lambda i: 64 - i,
                            lambda i: hex2ba("FFFFFFFFFFFFFFFF") >> i,
                            var_checks)
    
    # Run custom update to remove synapses on diagonal
    model.custom_update("RemoveSynapse")

    # Check resultant connectivity
    for pop, var_checks in samples.items():
        _check_connectivity(pop, lambda i: 0 if i > 63 else (63 - i),
                            lambda i: hex2ba("7FFFFFFFFFFFFFFF") >> i,
                            var_checks)

    # Run custom update to add diagonal synapses back again
    model.custom_update("AddSynapse")

    # Check resultant connectivity
    # **TODO** check a variable on remove_synapse_ccu
    _check_connectivity(s_pop_1, 
                        lambda i: 64 - i,
                        lambda i: hex2ba("FFFFFFFFFFFFFFFF") >> i,
                        [(s_pop_1, "g", False), 
                         (s_pop_1, "d", True)])


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_connectivity_update_delay(make_model, backend, precision):
    pre_neuron_model = create_neuron_model(
        "pre_neuron",
        sim_code=
        """
        removeIdx = (id + (int)round(t / dt)) % 64;
        """,
        vars=[("removeIdx", "int")])

    post_neuron_model = create_neuron_model(
        "post_neuron",
        sim_code=
        """
        remove = ((id + (int)round(t / dt)) % 64) == 0;
        """,
        vars=[("remove", "bool")])

    # Weight update model that does something arbitrary stuff with 
    # presynaptic variable reference to make sure it gets delayed
    pre_weight_update_model = create_weight_update_model(
        "pre_weight_update",
        pre_neuron_var_refs=[("removeIdx", "int", VarAccessMode.READ_ONLY)],
        vars=[("g", "scalar", VarAccess.READ_ONLY)],
        pre_spike_syn_code=
        """
        addToPost(g * (float)removeIdx);
        """)

    # Weight update model that does something arbitrary stuff with 
    # postsynaptic variable reference to make sure it gets delayed
    post_weight_update_model = create_weight_update_model(
        "post_weight_update",
        post_neuron_var_refs=[("remove", "bool", VarAccessMode.READ_ONLY)],
        vars=[("g", "scalar", VarAccess.READ_ONLY)],
        pre_spike_syn_code=
        """
        addToPost(g * (float)remove);
        """)

    # Snippet to configure a dense matrix of sparse connectivity
    dense_connect_init_snippet = create_sparse_connect_init_snippet(
        "dense_connect_init",
        row_build_code=
        """
        for(unsigned int j = 0; j < num_post; j++) {
            addSynapse(j);
        }
        """)

    remove_synapse_pre_model = create_custom_connectivity_update_model(
        "remove_synapse_pre",
        pre_var_refs=[("removeIdx", "int")],
        row_update_code=
        """
        for_each_synapse {
            if(id_post == removeIdx) {
                remove_synapse();
                break;
            }
        }
        """)
    remove_synapse_post_model = create_custom_connectivity_update_model(
        "remove_synapse_post",
        post_var_refs=[("remove", "bool")],
        row_update_code=
        """
        for_each_synapse {
            if(remove) {
                remove_synapse();
                break;
            }
        }
        """)

    model = make_model(precision, "test_custom_connectivity_update_delay", backend=backend)
    model.dt = 1.0
    
    # Create pre and postsynaptic populations
    pre_n_pop = model.add_neuron_population("PreNeurons", 64, pre_neuron_model, 
                                            {}, {"removeIdx": 0}); 
    post_n_pop = model.add_neuron_population("PostNeurons", 64, post_neuron_model, 
                                             {}, {"remove": False}); 

    # Create synapse groups
    s_pop_1 = model.add_synapse_population(
        "Syn1", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(pre_weight_update_model, {}, {"g": init_var(weight_init_snippet)},
                           pre_var_refs={"removeIdx": create_var_ref(pre_n_pop, "removeIdx")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(dense_connect_init_snippet))
    s_pop_1.axonal_delay_steps = 5

    s_pop_2 = model.add_synapse_population(
        "Syn2", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(post_weight_update_model, {}, {"g": init_var(weight_init_snippet)},
                           post_var_refs={"remove": create_var_ref(post_n_pop, "remove")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(dense_connect_init_snippet))
    s_pop_2.back_prop_delay_steps = 5

    model.add_custom_connectivity_update(
        "RemoveSynapsePre", "RemoveSynapse", s_pop_1,
        remove_synapse_pre_model,
        pre_var_refs={"removeIdx": create_var_ref(pre_n_pop, "removeIdx")})
    
    model.add_custom_connectivity_update(
        "RemoveSynapsePost", "RemoveSynapse", s_pop_2,
        remove_synapse_post_model,
        post_var_refs={"remove": create_var_ref(post_n_pop, "remove")})
    
    # Build model and load
    model.build()
    model.load()
        
    # Check initial connectivity
    dense_bitarray = hex2ba("FFFFFFFFFFFFFFFF")
    _check_connectivity(s_pop_1, lambda i: 64, lambda i: dense_bitarray,
                        [(s_pop_1, "g", False)])
    _check_connectivity(s_pop_2, lambda i: 64, lambda i: dense_bitarray,
                        [(s_pop_2, "g", False)])

    # Run for 5 timesteps
    while model.timestep < 5:
        model.step_time()
    

    # Run update to remove synapses
    model.custom_update("RemoveSynapse")

    # Check updated connectivity
    _check_connectivity(s_pop_1, lambda i: 63,
                        lambda i: _clear_bit(dense_bitarray, (i + 4) % 64),
                        [(s_pop_1, "g", False)])
    _check_connectivity(s_pop_2, lambda i: 63,
                        lambda i: _clear_bit(dense_bitarray, 60),
                        [(s_pop_2, "g", False)])

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_connectivity_update_remap(make_model, backend, precision):
    """
    for j in range(16):
        for i in range(4):
            i_value = 1 << i
            if ((j + 1) & i_value) != 0:
                pre_inds.append(i)
                post_inds.append(j)
    """
    decoder_model = create_sparse_connect_init_snippet(
        "decoder",
        row_build_code=
        """
        const unsigned int iValue = 1 << id_pre;
        for(unsigned int j = 0; j < num_post; j++) {
            if(((j + 1) & iValue) != 0) {
                addSynapse(j);
            }
        }
        """)
    pre_reverse_model = create_neuron_model(
        "pre_reverse",
        vars=[("x", "scalar")],
        sim_code=
        """
        x = Isyn;
        """)

    static_pulse_reverse_post_model = create_weight_update_model(
        "static_pulse_reverse_post",
        post_spike_syn_code=
        """
        addToPre(g);
        """,
        vars=[("g", "scalar", VarAccess.READ_ONLY)])
    
    remove_ones_model = create_custom_connectivity_update_model(
        "remove_synapse",
        row_update_code=
        """
        if(id_pre == 0) {
            for_each_synapse {
                remove_synapse();
            }
        }
        """)
        
    model = make_model(precision, "test_custom_connectivity_update_remap", backend=backend)
    model.dt = 1.0

    # Add presynaptic populations to sum reverse input
    sparse_pre_n_pop = model.add_neuron_population(
        "SparsePost", 4, pre_reverse_model,
        {}, {"x": 0.0})

    # Create spike source array to generate one-hot pattern to decode
    post_n_pop = model.add_neuron_population(
        "SpikeSource", 16, "SpikeSourceArray",
        {}, {"startSpike": np.arange(16), "endSpike": np.arange(1, 17)})
    post_n_pop.extra_global_params["spikeTimes"].set_init_values(np.arange(16.0))

    # Add connectivity
    sparse_s_pop = model.add_synapse_population(
        "SparseSynapse", "SPARSE",
        sparse_pre_n_pop, post_n_pop,
        init_weight_update(static_pulse_reverse_post_model, {}, {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(decoder_model))
    
    # Add custom update to remove a row
    model.add_custom_connectivity_update(
        "RemoveOnes", "RemoveSynapse", sparse_s_pop,
        remove_ones_model)

    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    output_place_values = 2 ** np.arange(4)
    while model.timestep < 16:
        model.step_time()
        
        # Pull state variable
        sparse_pre_n_pop.vars["x"].pull_from_device()

        # Convert to binary mask
        output_binary = np.isclose(np.ones(4), sparse_pre_n_pop.vars["x"].view)

        # Sum up active place values
        output_value = np.sum(output_place_values[output_binary])
        if output_value != (model.timestep - 1):
            assert False, f"{sparse_pre_n_pop.name} decoding incorrect ({output_value} rather than {model.timestep - 1})"

    # Run update to remove synapses
    model.custom_update("RemoveSynapse")

    # Reset time and start spike
    model.timestep = 0
    post_n_pop.vars["startSpike"].view[:] = np.arange(16)
    post_n_pop.vars["startSpike"].push_to_device()

    # Simulate another 16 timesteps
    while model.timestep < 16:
        model.step_time()
        
        # Pull state variable
        sparse_pre_n_pop.vars["x"].pull_from_device()

        # Convert to binary mask
        output_binary = np.isclose(np.ones(4), sparse_pre_n_pop.vars["x"].view)

        # Sum up active place values
        output_value = np.sum(output_place_values[output_binary])
        correct = (model.timestep - 1) & ~1
        if output_value != correct:
            assert False, f"{sparse_pre_n_pop.name} decoding incorrect ({output_value} rather than {model.timestep - 1})"
