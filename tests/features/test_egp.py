import numpy as np
import pytest
from pygenn import types

from pygenn import (create_current_source_model, 
                    create_custom_update_model,
                    create_egp_ref,
                    create_neuron_model,
                    create_postsynaptic_model,
                    create_var_init_snippet,
                    create_var_ref,
                    create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_weight_update, init_var)

# Neuron model which does nothing
empty_neuron_model = create_neuron_model("empty")

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_egp_var_init(make_model, backend, precision):
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
        vars=[("repeat", "scalar")])

    nop_current_source_model = create_current_source_model(
        "nop_current_source",
        vars=[("repeat", "scalar")])

    nop_postsynaptic_update_model = create_postsynaptic_model(
        "nop_postsynaptic_update",
        vars=[("psm_repeat", "scalar")])

    nop_weight_update_model = create_weight_update_model(
        "nop_weight_update",
        vars=[("repeat", "scalar")],
        pre_vars=[("pre_repeat", "scalar")],
        post_vars=[("post_repeat", "scalar")])
    
    model = make_model(precision, "test_egp_var_init", backend=backend)

    ss_pop = model.add_neuron_population("SpikeSource", 20, empty_neuron_model);
    
    # Create populations with randomly-initialised variables
    correct = np.arange(10.0)
    n_pop = model.add_neuron_population("Neurons", 20, nop_neuron_model, {}, {"repeat": init_var(repeat_var_init_snippet)})
    n_pop.vars["repeat"].extra_global_params["values"].set_init_values(correct)
    
    cs = model.add_current_source("CurrentSource", nop_current_source_model, n_pop,
                                  {}, {"repeat": init_var(repeat_var_init_snippet)})
    cs.vars["repeat"].extra_global_params["values"].set_init_values(correct)
    
    dense_s_pop = model.add_synapse_population(
        "DenseSynapses", "DENSE",
        ss_pop, n_pop,
        init_weight_update(nop_weight_update_model, {}, {"repeat": init_var(pre_repeat_var_init_snippet)}, {"pre_repeat": init_var(repeat_var_init_snippet)}, {"post_repeat": init_var(repeat_var_init_snippet)}),
        init_postsynaptic(nop_postsynaptic_update_model, {}, {"psm_repeat": init_var(repeat_var_init_snippet)}))
    dense_s_pop.vars["repeat"].extra_global_params["values"].set_init_values(correct)
    dense_s_pop.pre_vars["pre_repeat"].extra_global_params["values"].set_init_values(correct)
    dense_s_pop.post_vars["post_repeat"].extra_global_params["values"].set_init_values(correct)
    dense_s_pop.psm_vars["psm_repeat"].extra_global_params["values"].set_init_values(correct)
    
    sparse_s_pop = model.add_synapse_population(
        "SparseSynapses", "SPARSE",
        ss_pop, n_pop,
        init_weight_update(nop_weight_update_model, {}, {"repeat": init_var(post_repeat_var_init_snippet)}, {"pre_repeat": init_var(repeat_var_init_snippet)}, {"post_repeat": init_var(repeat_var_init_snippet)}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    sparse_s_pop.vars["repeat"].extra_global_params["values"].set_init_values(correct)
    sparse_s_pop.pre_vars["pre_repeat"].extra_global_params["values"].set_init_values(correct)
    sparse_s_pop.post_vars["post_repeat"].extra_global_params["values"].set_init_values(correct)

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
    dense_s_pop.vars["repeat"].pull_from_device()
    if not np.allclose(dense_s_pop.vars["repeat"].values, np.repeat(tiled_correct, 20)):
        assert False, f"'{dense_s_pop.name}' initialisation incorrect"
    
    # Download sparse connectivity
    sparse_s_pop.pull_connectivity_from_device()

    # Check sparse
    sparse_s_pop.vars["repeat"].pull_from_device()
    if not np.allclose(sparse_s_pop.vars["repeat"].values, tiled_correct):
        assert False, f"'{sparse_s_pop.name}' initialisation incorrect"

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_egp_ref(make_model, backend, precision):
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        x = e[id];
        """,
        vars=[("x", "scalar")],
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

    current_source_model = create_current_source_model(
        "current_source_model",
        injection_code=
        """
        if(id == (int)round(fmod(t, 10.0))) {
           e[id] = 1.0;
        }
        else {
           e[id] = 0.0;
        }
        """,
        neuron_extra_global_param_refs=[("e", "scalar*")])

    postsynaptic_model = create_postsynaptic_model(
        "postsynaptic_model",
        sim_code=
        """
        if(id == (int)round(fmod(t, 10.0))) {
           e[id] = 1.0;
        }
        else {
           e[id] = 0.0;
        }
        """,
        neuron_extra_global_param_refs=[("e", "scalar*")])

    model = make_model(precision, "test_egp_ref", backend=backend)
    model.dt = 1.0

    n_pop_cu = model.add_neuron_population("NeuronsCU", 10, neuron_model, {}, {"x": 10.0})
    n_pop_cu.extra_global_params["e"].set_init_values(np.empty(10))

    n_pop_cs = model.add_neuron_population("NeuronsCS", 10, neuron_model, {}, {"x": 10.0})
    n_pop_cs.extra_global_params["e"].set_init_values(np.empty(10))

    n_pop_psm = model.add_neuron_population("NeuronsPSM", 10, neuron_model, {}, {"x": 10.0})
    n_pop_psm.extra_global_params["e"].set_init_values(np.empty(10))

    model.add_custom_update("CU", "CustomUpdate", custom_update_model,
                            {}, {}, {"v": create_var_ref(n_pop_cu, "x")}, {"e": create_egp_ref(n_pop_cu, "e")})

    model.add_current_source("CS", current_source_model, n_pop_cs,
                             egp_refs={"e": create_egp_ref(n_pop_cs, "e")})
    
    model.add_synapse_population("SG", "DENSE", n_pop_cu, n_pop_psm,
                                 init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
                                 init_postsynaptic(postsynaptic_model, egp_refs={"e": create_egp_ref(n_pop_psm, "e")}))

    # Build model and load
    model.build()
    model.load()

    n_pops = [n_pop_cu, n_pop_cs, n_pop_psm]
    while model.timestep < 10:
        model.custom_update("CustomUpdate")
        model.step_time()

        correct = np.zeros(10)
        correct[model.timestep - 1] = 1.0

        for n in n_pops:
            n.vars["x"].pull_from_device()
            assert np.allclose(n.vars["x"].view, correct)
