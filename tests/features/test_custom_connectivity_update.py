import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel
from pygenn.genn import VarAccess, VarAccessMode

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
            break;\n"
        }
    }
    """)

"""
class RemoveSynapseHostEGPUpdate : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapseHostEGPUpdate, 0, 0, 0, 0, 0, 0, 0);
    
    SET_EXTRA_GLOBAL_PARAMS({{"d", "uint32_t*"}});
    SET_ROW_UPDATE_CODE(
        "const unsigned int wordsPerRow = ($(num_post) + 31) / 32;\n"
        "$(for_each_synapse,\n"
        "{\n"
        "   if($(d)[(wordsPerRow * $(id_pre)) + ($(id_post) / 32)] & (1 << ($(id_post) % 32))) {\n"
        "       $(remove_synapse);\n"
        "   }\n"
        "});\n");
    SET_HOST_UPDATE_CODE(
        "const unsigned int wordsPerRow = ($(num_post) + 31) / 32;\n"
        "memset($(d), 0, wordsPerRow * $(num_pre) * sizeof(uint32_t));\n"
        "for(unsigned int i = 0; i < $(num_pre); i++) {\n"
        "   uint32_t *dRow = &$(d)[wordsPerRow * i];\n"
        "   dRow[i / 32] |= (1 << (i % 32));\n"
        "}\n"
        "$(pushdToDevice, wordsPerRow * $(num_pre));\n");
};
IMPLEMENT_MODEL(RemoveSynapseHostEGPUpdate);

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
    }
    """)

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
    
    # Create custom connectivity updates
    remove_synapse_ccu = model.add_custom_connectivity_update(
        "RemoveSynapse", "RemoveSynapse", s_pop_1,
        remove_synapse_model,
        {}, {"a": init_var(weight_init_snippet)}, {}, {}, 
        {}, {}, {})

    add_synapse_ccu = model.add_custom_connectivity_update(
        "AddSynapse", "AddSynapse", s_pop_1,
        add_synapse_model,
        {}, {}, {}, {}, {},
        {"g": create_wu_var_ref(s_pop_1, "g"), "d": create_wu_var_ref(s_pop_1, "d"), "a": create_wu_var_ref(remove_synapse_ccu, "a")}, {}, {})    

    # Build model and load
    model.build()
    model.load()

if __name__ == '__main__':
    test_custom_connectivity_update("single_threaded_cpu", types.Float)
