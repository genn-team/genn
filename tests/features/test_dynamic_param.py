import numpy as np
import pytest
from pygenn import types

from pygenn import (create_current_source_model, 
                    create_custom_connectivity_update_model,
                    create_custom_update_model,
                    create_neuron_model,
                    create_postsynaptic_model,
                    create_var_init_snippet,
                    create_var_ref,
                    create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_weight_update, init_var)

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_dynamic_param(make_model, backend, precision):
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        x = t + shift + input;
        """,
        params=["input"],
        var_name_types=[("x", "scalar"), ("shift", "scalar")])
    
    postsynaptic_model = create_postsynaptic_model(
        "postsynaptic",
        sim_code=
        """
        injectCurrent(inSyn);
        psmX = t + psmShift + psmInput;
        $(inSyn) = 0;
        """,
        params=["psmInput"],
        var_name_types=[("psmX", "scalar"), ("psmShift", "scalar")])

    weight_update_model = create_weight_update_model(
        "weight_update",
        synapse_dynamics_code=
        """
        x = t + shift + input;
        """,
        params=["input"],
        var_name_types=[("x", "scalar"), ("shift", "scalar")])
    
    
    current_source_model = create_current_source_model(
        "current_source",
        injection_code=
        """
        x = t + shift + input;
        injectCurrent(0.0);
        """,
        params=["input"],
        var_name_types=[("x", "scalar"), ("shift", "scalar")])
    
    custom_update_model = create_custom_update_model(
        "custom_update",
        update_code=
        """
        x = t + shift + input;
        """,
        params=["input"],
        var_refs=[("y", "scalar")],
        var_name_types=[("x", "scalar"), ("shift", "scalar")])

    custom_connectivity_update_model = create_custom_connectivity_update_model(
        "custom_connectivity_update",
        row_update_code=
        """
        xDevice = t + shift + input;
        """,
        host_update_code=
        """
        pullshiftFromDevice();
        for(int i = 0; i < num_pre; i++) {
            xHost[i] = t + shift[i] + input;
        }
        pushxHostToDevice();
        """,
        params=["input"],
        pre_var_name_types=[("xDevice", "scalar"), ("xHost", "scalar"),
                            ("shift", "scalar")])

    model = make_model(precision, "test_dynamic_param", backend=backend)
    model.dt = 1.0
    
    shift = np.arange(0.0, 100.0, 10.0)
    pre_n_pop = model.add_neuron_population("PreNeurons", 10, neuron_model, 
                                            {"input": 0.0},
                                            {"x": 0.0, "shift": shift})
    pre_n_pop.set_param_dynamic("input")
    
    post_n_pop = model.add_neuron_population("PostNeurons", 10, neuron_model, 
                                            {"input": 0.0},
                                            {"x": 0.0, "shift": shift})
    
    cs_pop = model.add_current_source("CurrentSource", 
                                      current_source_model, pre_n_pop,
                                      {"input": 0.0},
                                      {"x": 0.0, "shift": shift})
    cs_pop.set_param_dynamic("input")

    cu = model.add_custom_update(
        "CustomUpdate", "CustomUpdate", custom_update_model,
        {"input": 0.0}, {"x": 0.0, "shift": shift}, 
        {"y": create_var_ref(pre_n_pop, "x")})
    cu.set_param_dynamic("input")

    s_pop = model.add_synapse_population(
        "Synapse", "SPARSE", 0,
        pre_n_pop, post_n_pop,
        init_weight_update(weight_update_model, {"input": 0.0}, {"x": 0.0, "shift": shift}),
        init_postsynaptic(postsynaptic_model, {"psmInput": 0.0}, {"psmX": 0.0, "psmShift": shift}))
    s_pop.set_sparse_connections(np.arange(10), np.arange(10))
    s_pop.set_wu_param_dynamic("input")
    s_pop.set_ps_param_dynamic("psmInput")

    ccu = model.add_custom_connectivity_update(
        "CustomConnectivityUpdate", "CustomUpdate", s_pop,
        custom_connectivity_update_model,
        {"input": 0.0}, {}, {"xDevice": 0.0, "xHost": 0.0, "shift": shift})
    ccu.set_param_dynamic("input")

    # Build model and load
    model.build()
    model.load()
    
    vars = [(pre_n_pop.vars["x"], "input"), (cs_pop.vars["x"], "input"),
            (cu.vars["x"], "input"), (s_pop.vars["x"], "input"), 
            (s_pop.psm_vars["psmX"], "psmInput"), 
            (ccu.pre_vars["xHost"], "input"), (ccu.pre_vars["xDevice"], "input")]
    while model.timestep < 10:
        correct = model.t + shift + (model.t ** 2)
        model.custom_update("CustomUpdate")
        model.step_time()
    
        # Loop through populations
        for v, p in vars:
            v.pull_from_device()
            values = v.values

            if not np.allclose(values, correct):
                assert False, f"{v.group.name} var {v.name} has wrong value ({values}) rather than {correct})"

            # Set dynamic parameter
            v.group.set_dynamic_param_value(p, model.t ** 2)
