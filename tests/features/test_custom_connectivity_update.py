import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel
from pygenn.genn import VarAccess, VarAccessMode

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
                    init_sparse_connectivity,
                    init_var)


neuron_model = create_neuron_model(
    "neuron",
    var_name_types=[("V", "scalar")])

weight_update_model = create_weight_update_model(
    "weight_update",
    var_name_types=[("g", "scalar", VarAccess.READ_ONLY),
                    ("d", "unsigned int", VarAccess.READ_ONLY)])


weight_init_snippet = create_var_init_snippet(
    "weight_init",
    var_init_code=
    """
    value = (id_pre * 64) + id_post;
    """)

delay_init_snippet = create_var_init_snippet(
    "delay_init",
    var_init_code=
    """
    value = (id_post * 64) + id_pre;
    """)

triangle_connect_init_snippet = create_sparse_connect_init_snippet(
    "triangle_connect_init",
    row_build_code=
    """
    for(unsigned int j = id_pre; j < num_post; j++) {
        addSynapse(j);
    }
    """)

remove_synapse_model = create_custom_connectivity_update_model(
    "remove_synapse",
    var_name_types=[("a", "scalar")],
    row_update_code=
    """
    for_each_synapse {
        if(id_post == id_pre) {
            remove_synapse();
            break;
        }
    }
    """)

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
        uint32_t *dRow = &d[wordsPerRow * i];
        dRow[i / 32] |= (1 << (i % 32));
    }
    pushdToDevice(wordsPerRow * num_pre);
    """)
    
"""
class RemoveSynapseHostVarUpdate : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapseHostVarUpdate, 0, 0, 1, 0, 0, 0, 0);
    
    SET_PRE_VARS({{"postInd", "unsigned int"}});
    SET_ROW_UPDATE_CODE(
        "$(for_each_synapse,\n"
        "{\n"
        "   if($(postInd) == $(id_post)) {\n"
        "       $(remove_synapse);\n"
        "       break;\n"
        "   }\n"
        "});\n");
    SET_HOST_UPDATE_CODE(
        "for(unsigned int i = 0; i < $(num_pre); i++) {\n"
        "   $(postInd)[i] = i;\n"
        "}\n"
        "$(pushpostIndToDevice);\n");
};
IMPLEMENT_MODEL(RemoveSynapseHostVarUpdate);
"""
add_synapse_model = create_custom_connectivity_update_model(
    "add_synapse",
    var_refs=[("g", "scalar"), ("d", "unsigned int"), ("a", "scalar")],
    row_update_code=
    """
    const scalar weight = (id_pre * 64) + id_pre;
    const unsigned int delay = (id_pre * 64) + id_pre;
    add_synapse(id_pre, weight, delay, weight);
    """)

def _check_connectivity(sg, get_row_length_fn, get_connectivity_fn, var_checks=[]):
    sg.pull_connectivity_from_device()

    # Pull all variables from device and get numpy array of values
    var_values = []
    for pop, var_name, _ in var_checks:
        pop.pull_var_from_device(var_name)
        var_values.append(pop.get_var_values(var_name))

    pre_inds = sg.get_sparse_pre_inds()
    post_inds = sg.get_sparse_post_inds()

    # Loop through rows
    row_lengths = np.bincount(pre_inds, minlength=sg.src.size)
    for i in range(sg.src.size):
        # Check row lengths
        assert row_lengths[i] == get_row_length_fn(i)

        # Build mask of row
        row_mask = (pre_inds == i)
        row_inds = post_inds[row_mask]

        # Build bitarray of row connectivity
        # **YUCK** converting to list
        row_bits = bitarray(sg.trg.size)
        row_bits.setall(0)
        row_bits[list(row_inds)] = 1
        
        # Check connectivity
        assert row_bits == get_connectivity_fn(i)

        # Loop through variables and check values match chosen pattern
        for (pop, var_name, transpose), val in zip(var_checks, var_values):
            correct = ((row_inds * 64.0) + i if transpose 
                       else (i * 64.0) + row_inds)
            assert np.allclose(val[row_mask], correct)



@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_custom_connectivity_update(backend, precision):
    model = GeNNModel(precision, "test_custom_connectivity_update", backend=backend)
    model.dt = 1.0
    
    # Create pre and postsynaptic populations
    pre_n_pop = model.add_neuron_population("PreNeurons", 64, "SpikeSource", {}, {}); 
    post_n_pop = model.add_neuron_population("PostNeurons", 64, "SpikeSource", {}, {}); 

    # Create synapse groups
    s_pop_1 = model.add_synapse_population(
        "Syn1", "SPARSE", 0,
        pre_n_pop, post_n_pop,
        weight_update_model, {}, {"g": init_var(weight_init_snippet), "d": init_var(delay_init_snippet)}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity(triangle_connect_init_snippet))

    s_pop_2 = model.add_synapse_population(
        "Syn2", "SPARSE", 0,
        pre_n_pop, post_n_pop,
        weight_update_model, {}, {"g": init_var(weight_init_snippet), "d": init_var(delay_init_snippet)}, {}, {},
        "DeltaCurr", {}, {},
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
    num_words = post_n_pop.size * ((pre_n_pop.size + 31) // 32)
    remove_synapse_host_egp_ccu.extra_global_params["d"].set_values(
        np.empty(num_words, dtype=np.uint32))

    add_synapse_ccu = model.add_custom_connectivity_update(
        "AddSynapse", "AddSynapse", s_pop_1,
        add_synapse_model,
        {}, {}, {}, {},
        {"g": create_wu_var_ref(s_pop_1, "g"), "d": create_wu_var_ref(s_pop_1, "d"), "a": create_wu_var_ref(remove_synapse_ccu, "a")}, {}, {})

    # Build model and load
    model.build()
    model.load()
    
    # **TODO** check a variable on remove_synapse_ccu
    samples = {s_pop_1: [(s_pop_1, "g", False), (s_pop_1, "d", True)],
               s_pop_2: [(s_pop_2, "g", False), (s_pop_2, "d", True)]}
                
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

if __name__ == '__main__':
    test_custom_connectivity_update("single_threaded_cpu", types.Float)
