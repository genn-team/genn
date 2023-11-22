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
                    create_var_ref,
                    create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_weight_update, init_var)

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_dynamic_param(backend, precision):
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        x = t + shift + input;
        """,
        params=["input"],
        var_name_types=[("x", "scalar"), ("shift", "scalar")])
    
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

    model = GeNNModel(precision, "test_dynamic_param", backend=backend)
    model.dt = 1.0
    
    shift = np.arange(0.0, 100.0, 10.0)
    pre_n_pop = model.add_neuron_population("PreNeurons", 10, neuron_model, 
                                            {"input": 0.0},
                                            {"x": 0.0, "shift": shift})
    pre_n_pop.set_param_dynamic("input")

    post_n_pop = model.add_neuron_population("PostNeurons", 1, neuron_model, 
                                            {"input": 0.0},
                                            {"x": 0.0, "shift": 0})

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
        init_postsynaptic("DeltaCurr"))
    s_pop.set_sparse_connections(np.arange(10), np.arange(10))
    s_pop.set_wu_param_dynamic("input")

    ccu = model.add_custom_connectivity_update(
        "CustomConnectivityUpdate", "CustomUpdate", s_pop,
        custom_connectivity_update_model,
        {"input": 0.0}, {}, {"xDevice": 0.0, "xHost": 0.0, "shift": shift})
    ccu.set_param_dynamic("input")

    # Build model and load
    model.build()
    model.load()
    
    vars = [pre_n_pop.vars["x"], cs_pop.vars["x"], 
            cu.vars["x"], s_pop.vars["x"], ccu.pre_vars["xHost"],
            ccu.pre_vars["xDevice"]]
    while model.timestep < 10:
        correct = model.t + shift + (model.t ** 2)
        model.custom_update("CustomUpdate")
        model.step_time()
    
        # Loop through populations
        for v in vars:
            v.pull_from_device()
            values = v.values
            if not np.allclose(values, correct):
                assert False, f"{v.group.name} var x has wrong value ({values}) rather than {correct})"

            # Set dynamic parameter
            v.group.set_dynamic_param_value("input", model.t ** 2)
        

if __name__ == '__main__':
    test_dynamic_param("cuda", types.Float)
