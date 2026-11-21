import numpy as np
import pytest
from pygenn import types

from pygenn import (create_current_source_model,
                    create_custom_connectivity_update_model,
                    create_custom_update_model,
                    create_neuron_model,
                    create_postsynaptic_model,
                    create_sparse_connect_init_snippet,
                    create_var_init_snippet,
                    create_var_ref,
                    create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_weight_update, init_var)

# Timestep used by all models - not 1.0 so that
# derived parameters which depend on dt can be distinguished
DT = 0.5

def _calc_derived(input, dt=DT):
    return (2.0 * input) + (10.0 * dt)

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_derived_param(make_model, backend, precision):
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        x = t + shift + inputDerived;
        """,
        params=["input"],
        derived_params=[("inputDerived", lambda pars, dt: _calc_derived(pars["input"], dt))],
        vars=[("x", "scalar"), ("shift", "scalar")])

    postsynaptic_model = create_postsynaptic_model(
        "postsynaptic",
        sim_code=
        """
        injectCurrent(inSyn);
        psmX = t + psmShift + psmInputDerived;
        inSyn = 0;
        """,
        params=["psmInput"],
        derived_params=[("psmInputDerived", lambda pars, dt: _calc_derived(pars["psmInput"], dt))],
        vars=[("psmX", "scalar"), ("psmShift", "scalar")])

    weight_update_model = create_weight_update_model(
        "weight_update",
        synapse_dynamics_code=
        """
        x = t + shift + inputDerived;
        """,
        pre_dynamics_code=
        """
        preX = t + preShift + inputDerived;
        """,
        post_dynamics_code=
        """
        postX = t + postShift + inputDerived;
        """,
        params=["input"],
        derived_params=[("inputDerived", lambda pars, dt: _calc_derived(pars["input"], dt))],
        vars=[("x", "scalar"), ("shift", "scalar")],
        pre_vars=[("preX", "scalar"), ("preShift", "scalar")],
        post_vars=[("postX", "scalar"), ("postShift", "scalar")])

    current_source_model = create_current_source_model(
        "current_source",
        injection_code=
        """
        x = t + shift + inputDerived;
        injectCurrent(0.0);
        """,
        params=["input"],
        derived_params=[("inputDerived", lambda pars, dt: _calc_derived(pars["input"], dt))],
        vars=[("x", "scalar"), ("shift", "scalar")])

    custom_update_model = create_custom_update_model(
        "custom_update",
        update_code=
        """
        x = t + shift + inputDerived;
        """,
        params=["input"],
        derived_params=[("inputDerived", lambda pars, dt: _calc_derived(pars["input"], dt))],
        var_refs=[("y", "scalar")],
        vars=[("x", "scalar"), ("shift", "scalar")])

    custom_connectivity_update_model = create_custom_connectivity_update_model(
        "custom_connectivity_update",
        row_update_code=
        """
        xDevice = t + shift + inputDerived;
        """,
        host_update_code=
        """
        pullshiftFromDevice();
        for(int i = 0; i < num_pre; i++) {
            xHost[i] = t + shift[i] + inputDerived;
        }
        pushxHostToDevice();
        """,
        params=["input"],
        derived_params=[("inputDerived", lambda pars, dt: _calc_derived(pars["input"], dt))],
        pre_vars=[("xDevice", "scalar"), ("xHost", "scalar"),
                  ("shift", "scalar")])

    # Variable initialisation snippet which uses derived parameter
    var_init_snippet = create_var_init_snippet(
        "var_init",
        var_init_code=
        """
        value = inputDerived + id;
        """,
        params=["input"],
        derived_params=[("inputDerived", lambda pars, dt: _calc_derived(pars["input"], dt))])

    # Sparse connectivity initialisation snippet which uses derived
    # parameter to connect each presynaptic neuron to a shifted postsynaptic neuron
    shifted_one_to_one_snippet = create_sparse_connect_init_snippet(
        "shifted_one_to_one",
        row_build_code=
        """
        addSynapse((id_pre + (unsigned int)offsetDerived) % num_post);
        """,
        params=["offset"],
        derived_params=[("offsetDerived", lambda pars, dt: pars["offset"] + 1.0)],
        calc_max_row_len_func=lambda num_pre, num_post, pars: 1,
        calc_max_col_len_func=lambda num_pre, num_post, pars: 1)

    model = make_model(precision, "test_derived_param", backend=backend)
    model.dt = DT

    # **NOTE** pre and postsynaptic populations use the same neuron model
    # with different parameter values to test heterogeneous derived parameters
    shift = np.arange(0.0, 100.0, 10.0)
    pre_n_pop = model.add_neuron_population("PreNeurons", 10, neuron_model,
                                            {"input": 1.0},
                                            {"x": 0.0, "shift": shift})

    post_n_pop = model.add_neuron_population("PostNeurons", 10, neuron_model,
                                             {"input": 2.0},
                                             {"x": 0.0, "shift": shift})

    # Add two structurally identical populations with different parameter
    # values so that they are merged and derived parameter is heterogeneous
    hetero_n_pop_1 = model.add_neuron_population("HeteroNeurons1", 10, neuron_model,
                                                 {"input": 8.0},
                                                 {"x": 0.0, "shift": shift})
    hetero_n_pop_2 = model.add_neuron_population("HeteroNeurons2", 10, neuron_model,
                                                 {"input": 9.0},
                                                 {"x": 0.0, "shift": shift})

    cs_pop = model.add_current_source("CurrentSource",
                                      current_source_model, pre_n_pop,
                                      {"input": 3.0},
                                      {"x": 0.0, "shift": shift})

    cu = model.add_custom_update(
        "CustomUpdate", "CustomUpdate", custom_update_model,
        {"input": 4.0},
        {"x": init_var(var_init_snippet, {"input": 10.0}), "shift": shift},
        {"y": create_var_ref(pre_n_pop, "x")})

    s_pop = model.add_synapse_population(
        "Synapse", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update(weight_update_model, {"input": 5.0},
                           {"x": 0.0, "shift": shift},
                           {"preX": 0.0, "preShift": shift},
                           {"postX": 0.0, "postShift": shift}),
        init_postsynaptic(postsynaptic_model, {"psmInput": 6.0},
                          {"psmX": 0.0, "psmShift": shift}))
    s_pop.set_sparse_connections(np.arange(10), np.arange(10))

    ccu = model.add_custom_connectivity_update(
        "CustomConnectivityUpdate", "CustomUpdate", s_pop,
        custom_connectivity_update_model,
        {"input": 7.0}, {}, {"xDevice": 0.0, "xHost": 0.0, "shift": shift})

    # Add second synapse population with connectivity
    # initialised using snippet with derived parameter
    shifted_s_pop = model.add_synapse_population(
        "ShiftedSynapse", "SPARSE",
        pre_n_pop, post_n_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(shifted_one_to_one_snippet, {"offset": 2.0}))

    # Build model and load
    model.build()
    model.load()

    # Check variable initialised using snippet with derived parameter
    cu.vars["x"].pull_from_device()
    assert np.allclose(cu.vars["x"].values, _calc_derived(10.0) + np.arange(10))

    # Check connectivity initialised using snippet with derived parameter
    shifted_s_pop.pull_connectivity_from_device()
    assert np.array_equal(shifted_s_pop.get_sparse_pre_inds(), np.arange(10))
    assert np.array_equal(shifted_s_pop.get_sparse_post_inds(),
                          (np.arange(10) + 3) % 10)

    vars = [(pre_n_pop.vars["x"], 1.0), (post_n_pop.vars["x"], 2.0),
            (hetero_n_pop_1.vars["x"], 8.0), (hetero_n_pop_2.vars["x"], 9.0),
            (cs_pop.vars["x"], 3.0), (cu.vars["x"], 4.0),
            (s_pop.vars["x"], 5.0), (s_pop.pre_vars["preX"], 5.0),
            (s_pop.post_vars["postX"], 5.0), (s_pop.psm_vars["psmX"], 6.0),
            (ccu.pre_vars["xHost"], 7.0), (ccu.pre_vars["xDevice"], 7.0)]
    while model.timestep < 10:
        t = model.t
        model.custom_update("CustomUpdate")
        model.step_time()

        # Loop through populations
        for v, p in vars:
            v.pull_from_device()
            values = v.values

            correct = t + shift + _calc_derived(p)
            if not np.allclose(values, correct):
                assert False, f"{v.group.name} var {v.name} has wrong value ({values}) rather than {correct})"
