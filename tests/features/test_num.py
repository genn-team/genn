import numpy as np
import pytest
from pygenn import types

from pygenn import VarAccessMode

from pygenn import (create_current_source_model, 
                    create_custom_update_model,
                    create_neuron_model,
                    create_postsynaptic_model,
                    create_var_init_snippet,
                    create_var_ref,
                    create_psm_var_ref,
                    create_weight_update_model,
                    create_wu_var_ref,
                    create_wu_pre_var_ref,
                    create_wu_post_var_ref,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_toeplitz_connectivity,
                    init_weight_update, init_var)

# Neuron model which does nothing
empty_neuron_model = create_neuron_model("empty")

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_num(make_model, backend, precision, batch_size):
    # Models which set state variables to double one of the num_XXX variables
    neuron_model = create_neuron_model(
        "neuron",
        vars=[("num_neurons_test", "unsigned int"),
                        ("num_batch_test", "unsigned int")],
        sim_code=
        """
        num_neurons_test = num_neurons * 2;
        num_batch_test = num_batch * 2;
        """)

    current_source_model = create_current_source_model(
        "current_source",
        vars=[("num_neurons_test", "unsigned int"),
                        ("num_batch_test", "unsigned int")],
        injection_code=
        """
        num_neurons_test = num_neurons * 2;
        num_batch_test = num_batch * 2;
        """)

    weight_update_model = create_weight_update_model(
        "weight_update",
        vars=[("num_pre_syn_test", "unsigned int"),
                        ("num_post_syn_test", "unsigned int"),
                        ("num_batch_syn_test", "unsigned int")],
        pre_vars=[("num_neurons_pre_test", "unsigned int"),
                            ("num_batch_pre_test", "unsigned int")],
        post_vars=[("num_neurons_post_test", "unsigned int"),
                             ("num_batch_post_test", "unsigned int")],
        
        synapse_dynamics_code=
        """
        num_pre_syn_test = num_pre * 2;
        num_post_syn_test = num_post * 2;
        num_batch_syn_test = num_batch * 2;
        """,
        pre_dynamics_code=
        """
        num_neurons_pre_test = num_neurons * 2;
        num_batch_pre_test = num_batch * 2;
        """,
        post_dynamics_code=
        """
        num_neurons_post_test = num_neurons * 2;
        num_batch_post_test = num_batch * 2;
        """)

    postsynaptic_update_model = create_postsynaptic_model(
        "postsynaptic_update",
        vars=[("num_neurons_test", "unsigned int"),
                        ("num_batch_test", "unsigned int")],
        sim_code=
        """
        num_neurons_test = num_neurons * 2;
        num_batch_test = num_batch * 2;
        """)

    custom_update_model = create_custom_update_model(
        "custom_update",
        vars=[("num_neurons_test", "unsigned int"),
                        ("num_batch_test", "unsigned int")],
        var_refs=[("ref", "unsigned int", VarAccessMode.READ_ONLY)],
        update_code=
        """
        num_neurons_test = num_neurons * 2;
        num_batch_test = num_batch * 2;
        """)

    custom_update_wu_model = create_custom_update_model(
        "custom_update_wu",
        vars=[("num_pre_syn_test", "unsigned int"),
                        ("num_post_syn_test", "unsigned int"),
                        ("num_batch_syn_test", "unsigned int")],
        var_refs=[("ref", "unsigned int", VarAccessMode.READ_ONLY)],
        update_code=
        """
        num_pre_syn_test = num_pre * 2;
        num_post_syn_test = num_post * 2;
        num_batch_syn_test = num_batch * 2;
        """)

    # Snippets to initialise variables to num_XXX variables
    num_neurons_snippet = create_var_init_snippet(
        "num_neurons",
        var_init_code=
        """
        value = num_neurons;
        """)

    num_pre_snippet = create_var_init_snippet(
        "num_pre",
        var_init_code=
        """
        value = num_pre;
        """)

    num_post_snippet = create_var_init_snippet(
        "num_post",
        var_init_code=
        """
        value = num_post;
        """)
    
    num_batch_snippet = create_var_init_snippet(
        "num_batch",
        var_init_code=
        """
        value = num_batch;
        """)

    model = make_model(precision, "test_num", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size

    # Create a variety of models
    neuron_var_init = {"num_neurons_test": init_var(num_neurons_snippet),
                       "num_batch_test": init_var(num_batch_snippet)}
    synapse_var_init = {"num_pre_syn_test": init_var(num_pre_snippet), 
                        "num_post_syn_test": init_var(num_post_snippet),
                        "num_batch_syn_test": init_var(num_batch_snippet)}
    ss_pop = model.add_neuron_population("Pre", 2, empty_neuron_model);
    n_pop = model.add_neuron_population("Post", 4, neuron_model, 
                                        {}, neuron_var_init)
    cs = model.add_current_source("CurrentSource", current_source_model, n_pop,
                                  {}, neuron_var_init)

    pre_var_init = {"num_neurons_pre_test": init_var(num_neurons_snippet),
                    "num_batch_pre_test": init_var(num_batch_snippet)}
    post_var_init = {"num_neurons_post_test": init_var(num_neurons_snippet),
                     "num_batch_post_test": init_var(num_batch_snippet)}
    syn = model.add_synapse_population(
        "Syn", "DENSE",
        ss_pop, n_pop,
        init_weight_update(weight_update_model, {}, synapse_var_init, pre_var_init, post_var_init),
        init_postsynaptic(postsynaptic_update_model, {}, neuron_var_init))

    var_refs = {"ref": create_var_ref(n_pop, "num_neurons_test")}
    cu = model.add_custom_update("CU", "Test", custom_update_model,
                                 {}, neuron_var_init, var_refs)

    wu_var_refs = {"ref": create_wu_var_ref(syn, "num_pre_syn_test")}
    cuw = model.add_custom_update("CUW", "Test", custom_update_wu_model,
                                  {}, synapse_var_init, wu_var_refs)

    # Build model and load
    model.build()
    model.load()

    # List of variables to check
    vars = [(n_pop.vars["num_neurons_test"], 4),
            (n_pop.vars["num_batch_test"], batch_size),
            (cs.vars["num_neurons_test"], 4),
            (cs.vars["num_batch_test"], batch_size),
            (syn.vars["num_pre_syn_test"], 2),
            (syn.vars["num_post_syn_test"], 4),
            (syn.vars["num_batch_syn_test"], batch_size),
            (syn.pre_vars["num_neurons_pre_test"], 2),
            (syn.pre_vars["num_batch_pre_test"], batch_size),
            (syn.post_vars["num_neurons_post_test"], 4),
            (syn.post_vars["num_batch_post_test"], batch_size),
            (syn.psm_vars["num_neurons_test"], 4),
            (syn.psm_vars["num_batch_test"], batch_size),
            (cu.vars["num_neurons_test"], 4),
            (cu.vars["num_batch_test"], batch_size),
            (cuw.vars["num_pre_syn_test"], 2),
            (cuw.vars["num_post_syn_test"], 4),
            (cuw.vars["num_batch_syn_test"], batch_size)]

    # Check variables after initialisation
    for v, val in vars:
        v.pull_from_device()
        
        # Check p-value exceed our confidence internal
        assert np.allclose(
            v.values, val
        ), f"'{v.group.name}' '{v.name}' has wrong value ({v.values} rather than {val})"

    # Simulate one timestep
    model.step_time()
    model.custom_update("Test")

    # Check again after update
    for v, val in vars:
        v.pull_from_device()
        
        # Check variables are now updated to DOUBLE value
        assert np.allclose(
            v.values, val * 2
        ), f"'{v.group.name}' '{v.name}' has wrong value ({v.values} rather than {val * 2})"