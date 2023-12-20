import pytest
from pygenn import types

from pygenn import VarAccessMode

from pygenn import (create_current_source_model, 
                    create_custom_update_model,
                    create_neuron_model,
                    create_postsynaptic_model,
                    create_weight_update_model,
                    create_var_ref,
                    create_psm_var_ref,
                    create_wu_var_ref,
                    create_wu_pre_var_ref,
                    create_wu_post_var_ref,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_toeplitz_connectivity,
                    init_weight_update, init_var)

@pytest.mark.parametrize("backend, batch_size", [("single_threaded_cpu", 1), 
                                                 ("cuda", 1), ("cuda", 5)])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_num_sim(make_model, backend, precision, batch_size):
    neuron_model = create_neuron_model(
        "neuron",
        var_name_types=[("num_neurons_test", "unsigned int"),
                        ("num_batch_test", "unsigned int")],
        sim_code=
        """
        num_neurons_test = num_neurons;
        num_batch_test = num_batch;
        """)

    current_source_model = create_current_source_model(
        "current_source",
        var_name_types=[("num_neurons_test", "unsigned int"),
                        ("num_batch_test", "unsigned int")],
        injection_code=
        """
        num_neurons_test = num_neurons;
        num_batch_test = num_batch;
        """)

    weight_update_model = create_weight_update_model(
        "weight_update",
        var_name_types=[("num_pre_syn_test", "unsigned int"),
                        ("num_post_syn_test", "unsigned int"),
                        ("num_batch_syn_test", "unsigned int")],
        pre_var_name_types=[("num_neurons_pre_test", "unsigned int"),
                            ("num_batch_pre_test", "unsigned int")],
        post_var_name_types=[("num_neurons_post_test", "unsigned int"),
                             ("num_batch_post_test", "unsigned int")],
        
        synapse_dynamics_code=
        """
        num_pre_syn_test = num_pre;
        num_post_syn_test = num_post;
        num_batch_syn_test = num_batch;
        """,
        pre_dynamics_code=
        """
        num_neurons_pre_test = num_neurons;
        num_batch_pre_test = num_batch;
        """,
        post_dynamics_code=
        """
        num_neurons_post_test = num_neurons;
        num_batch_post_test = num_batch;
        """)
        
    postsynaptic_update_model = create_postsynaptic_model(
        "postsynaptic_update",
        var_name_types=[("num_neurons_test", "unsigned int"),
                        ("num_batch_test", "unsigned int")],
        sim_code=
        """
        num_neurons_test = num_neurons;
        num_batch_test = num_batch;
        """)

    custom_update_model = create_custom_update_model(
        "custom_update",
        var_name_types=[("num_neurons_test", "unsigned int"),
                        ("num_batch_test", "unsigned int")],
        var_refs=[("ref", "unsigned int", VarAccessMode.READ_ONLY)],
        update_code=
        """
        num_neurons_test = num_neurons;
        num_batch_test = num_batch;
        """)
    
    custom_update_wu_model = create_custom_update_model(
        "custom_update_wu",
        var_name_types=[("num_pre_syn_test", "unsigned int"),
                        ("num_post_syn_test", "unsigned int"),
                        ("num_batch_syn_test", "unsigned int")],
        var_refs=[("ref", "unsigned int", VarAccessMode.READ_ONLY)],
        update_code=
        """
        num_pre_syn_test = num_pre;
        num_post_syn_test = num_post;
        num_batch_syn_test = num_batch;
        """)
    
    model = make_model(precision, "test_num_sim", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size
    
    # Create a variety of models
    neuron_var_init = {"num_neurons_test": 0, "num_batch_test": 0}
    synapse_var_init = {"num_pre_syn_test": 0, "num_post_syn_test": 0,
                        "num_batch_syn_test": 0}
    ss_pop = model.add_neuron_population("Pre", 2, "SpikeSource", {}, {});
    n_pop = model.add_neuron_population("Post", 4, neuron_model, 
                                        {}, neuron_var_init)
    cs = model.add_current_source("CurrentSource", current_source_model, n_pop,
                                  {}, neuron_var_init)
    
    pre_var_init = {"num_neurons_pre_test": 0, "num_batch_pre_test": 0}
    post_var_init = {"num_neurons_post_test": 0, "num_batch_post_test": 0}
    syn = model.add_synapse_population(
        "Syn", "DENSE", 0,
        ss_pop, n_pop,
        init_weight_update(weight_update_model, {}, synapse_var_init, pre_var_init, post_var_init),
        init_postsynaptic(postsynaptic_update_model, {}, neuron_var_init))
    
    # Build model and load
    model.build()
    model.load()