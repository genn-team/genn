import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel

from pygenn import (create_current_source_model, 
                    create_custom_connectivity_update_model,
                    create_custom_update_model,
                    create_neuron_model,
                    create_postsynaptic_model,
                    create_var_init_snippet,
                    create_weight_update_model,
                    init_sparse_connectivity,
                    init_var)

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_egp_var_init(backend, precision):
    # Create var init snippet which fills variable with 10 repeated value
    repeat_var_init_snippet = create_var_init_snippet(
        "repeat_var_init",
        var_init_code=
        """
        value = values[id % 10];
        """,
        extra_global_params=[("values", "scalar*")])
    
    # Create var init snippet which fills variable with 10 repeated value
    pre_repeat_var_init_snippet = create_var_init_snippet(
        "pre_repeat_var_init",
        var_init_code=
        """
        value = values[id_pre % 10];
        """,
        extra_global_params=[("values", "scalar*")])
        
    # Create var init snippet which fills variable with 10 repeated value
    post_repeat_var_init_snippet = create_var_init_snippet(
        "post_repeat_var_init",
        var_init_code=
        """
        value = values[id_post % 10];
        """,
        extra_global_params=[("values", "scalar*")])
    
    nop_neuron_model = create_neuron_model(
        "nop_neuron",
        var_name_types=[("repeat", "scalar")])

    nop_current_source_model = create_current_source_model(
        "nop_current_source",
        var_name_types=[("repeat", "scalar")])

    nop_postsynaptic_update_model = create_postsynaptic_model(
        "nop_postsynaptic_update",
        var_name_types=[("psm_repeat", "scalar")])

    nop_weight_update_model = create_weight_update_model(
        "nop_weight_update",
        var_name_types=[("repeat", "scalar")],
        pre_var_name_types=[("pre_repeat", "scalar")],
        post_var_name_types=[("post_repeat", "scalar")])
    
    model = GeNNModel(precision, "test_egp_var_init", backend=backend)

    ss1_pop = model.add_neuron_population("SpikeSource1", 1, "SpikeSource", {}, {});
    ss2_pop = model.add_neuron_population("SpikeSource2", 50000, "SpikeSource", {}, {});
    
    # Create populations with randomly-initialised variables
    n_pop = model.add_neuron_population("Neurons", 50000, nop_neuron_model, {}, {"repeat": init_var(repeat_var_init_snippet)})
    cs = model.add_current_source("CurrentSource", nop_current_source_model, n_pop,
                                  {}, {"repeat": init_var(repeat_var_init_snippet)})

    dense_s_pop = model.add_synapse_population(
        "DenseSynapses", "DENSE", 0,
        ss1_pop, n_pop,
        nop_weight_update_model, {}, {"repeat": init_var(repeat_var_init_snippet)}, {"pre_repeat": init_var(pre_repeat_var_init_snippet)}, {"post_repeat": init_var(post_repeat_var_init_snippet)},
        nop_postsynaptic_update_model, {}, {"psm_repeat": init_var(repeat_var_init_snippet)})
        
    sparse_s_pop = model.add_synapse_population(
        "SparseSynapses", "SPARSE", 0,
        ss2_pop, n_pop,
        nop_weight_update_model, {}, {"repeat": init_var(repeat_var_init_snippet)}, {"pre_repeat": init_var(pre_repeat_var_init_snippet)}, {"post_repeat": init_var(post_repeat_var_init_snippet)},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
    
    # Build model and load
    model.build()
    model.load()

if __name__ == '__main__':
    test_egp_var_init("single_threaded_cpu", types.Float)
