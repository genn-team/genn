import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel

from pygenn import (create_current_source_model, 
                    create_custom_connectivity_update_model,
                    create_custom_update_model,
                    create_egp_ref,
                    create_neuron_model,
                    create_postsynaptic_model,
                    create_var_init_snippet,
                    create_var_ref,
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

    ss_pop = model.add_neuron_population("SpikeSource", 20, "SpikeSource", {}, {});
    
    # Create populations with randomly-initialised variables
    correct = np.arange(10.0)
    n_pop = model.add_neuron_population("Neurons", 20, nop_neuron_model, {}, {"repeat": init_var(repeat_var_init_snippet)})
    n_pop.vars["repeat"].extra_global_params["values"].set_values(correct)
    
    cs = model.add_current_source("CurrentSource", nop_current_source_model, n_pop,
                                  {}, {"repeat": init_var(repeat_var_init_snippet)})
    cs.vars["repeat"].extra_global_params["values"].set_values(correct)
    
    dense_s_pop = model.add_synapse_population(
        "DenseSynapses", "DENSE", 0,
        ss_pop, n_pop,
        nop_weight_update_model, {}, {"repeat": init_var(pre_repeat_var_init_snippet)}, {"pre_repeat": init_var(repeat_var_init_snippet)}, {"post_repeat": init_var(repeat_var_init_snippet)},
        nop_postsynaptic_update_model, {}, {"psm_repeat": init_var(repeat_var_init_snippet)})
    dense_s_pop.vars["repeat"].extra_global_params["values"].set_values(correct)
    dense_s_pop.pre_vars["pre_repeat"].extra_global_params["values"].set_values(correct)
    dense_s_pop.post_vars["post_repeat"].extra_global_params["values"].set_values(correct)
    dense_s_pop.psm_vars["psm_repeat"].extra_global_params["values"].set_values(correct)
    
    sparse_s_pop = model.add_synapse_population(
        "SparseSynapses", "SPARSE", 0,
        ss_pop, n_pop,
        nop_weight_update_model, {}, {"repeat": init_var(post_repeat_var_init_snippet)}, {"pre_repeat": init_var(repeat_var_init_snippet)}, {"post_repeat": init_var(repeat_var_init_snippet)},
        "DeltaCurr", {}, {},
        init_sparse_connectivity("OneToOne"))
    sparse_s_pop.vars["repeat"].extra_global_params["values"].set_values(correct)
    sparse_s_pop.pre_vars["pre_repeat"].extra_global_params["values"].set_values(correct)
    sparse_s_pop.post_vars["post_repeat"].extra_global_params["values"].set_values(correct)

    # Build model and load
    model.build()
    model.load()

    # Loop through populations
    tiled_correct = np.tile(correct, 2)
    vars = [(n_pop, "", n_pop.vars), 
            (cs, "", cs.vars),
            (dense_s_pop, "pre_", dense_s_pop.pre_vars), 
            (dense_s_pop, "post_", dense_s_pop.post_vars),
            (dense_s_pop, "psm_", dense_s_pop.psm_vars,),
            (sparse_s_pop, "pre_", dense_s_pop.pre_vars), 
            (sparse_s_pop, "post_", dense_s_pop.post_vars)]
    for pop, prefix, var_dict in vars:
        # Pull var from devices
        var_dict[prefix + "repeat"].pull_from_device()

        # If distribution is discrete
        view = var_dict[prefix +  "repeat"].view
        
        if not np.allclose(view, tiled_correct):
            assert False, f"'{pop.name}' initialisation incorrect"
    
    # Check dense
    dense_s_pop.pull_var_from_device("repeat")
    if not np.allclose(dense_s_pop.get_var_values("repeat"), np.repeat(tiled_correct, 20)):
        assert False, f"'{dense_s_pop.name}' initialisation incorrect"
    
    # Download sparse connectivity
    sparse_s_pop.pull_connectivity_from_device()

    # Check sparse
    sparse_s_pop.pull_var_from_device("repeat")
    if not np.allclose(sparse_s_pop.get_var_values("repeat"), tiled_correct):
        assert False, f"'{sparse_s_pop.name}' initialisation incorrect"

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_egp_ref(backend, precision):
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        x = e[id];
        """,
        var_name_types=[("x", "scalar")],
        extra_global_params=[("e", "scalar*")])
    
    custom_update_model = create_custom_update_model(
        "custom_update",
        update_code=
        """
        if(id == (int)round(fmod(t, 10.0))) {
           e[id] = 1.0;
        }
        else {
           e[id] = 0.0;
        }
        """,
        var_refs=[("v", "scalar")],
        extra_global_param_refs=[("e", "scalar*")])
    
    model = GeNNModel(precision, "test_egp_ref", backend=backend)
    model.dt = 1.0
    
    n_pop = model.add_neuron_population("Neurons", 10, neuron_model, {}, {"x": 10.0});
    n_pop.extra_global_params["e"].set_values(np.empty(10))

    cu = model.add_custom_update("CU", "CustomUpdate", custom_update_model,
                                 {}, {}, {"v": create_var_ref(n_pop, "x")}, {"e": create_egp_ref(n_pop, "e")})

    # Build model and load
    model.build()
    model.load()
    
    while model.timestep < 10:
        model.custom_update("CustomUpdate")
        model.step_time()
        
        correct = np.zeros(10)
        correct[model.timestep - 1] = 1.0
        
        n_pop.vars["x"].pull_from_device()
        assert np.allclose(n_pop.vars["x"].view, correct)

if __name__ == '__main__':
    test_egp_var_init("cuda", types.Float)
