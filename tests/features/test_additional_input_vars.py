import numpy as np
import pytest
from pygenn import types

from pygenn import VarAccess, VarAccessMode
from pygenn import (create_current_source_model,
                    create_neuron_model,
                    create_var_ref,
                    create_weight_update_model,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_weight_update)

# Initial values of additional input variables
ISYN_A_INIT = 5.0
ISYN_B_INIT = -3.0

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_additional_input_vars(make_model, backend, precision):
    # Neuron model which copies standard and additional
    # input variables into state variables so they can be read
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        isyn = Isyn;
        isynA = IsynA;
        isynB = IsynB;
        """,
        vars=[("x", "scalar", VarAccess.READ_ONLY), ("isyn", "scalar"),
              ("isynA", "scalar"), ("isynB", "scalar")],
        additional_input_vars=[("IsynA", "scalar", ISYN_A_INIT),
                               ("IsynB", "scalar", ISYN_B_INIT)])

    # Weight update model which continuously delivers
    # presynaptic variable to postsynaptic neuron
    continuous_weight_update_model = create_weight_update_model(
        "continuous_weight_update",
        params=["g"],
        pre_neuron_var_refs=[("x", "scalar", VarAccessMode.READ_ONLY)],
        synapse_dynamics_code=
        """
        addToPost(g * x);
        """)

    # Weight update model which continuously delivers
    # postsynaptic variable to presynaptic neuron
    continuous_reverse_weight_update_model = create_weight_update_model(
        "continuous_reverse_weight_update",
        params=["g"],
        post_neuron_var_refs=[("x", "scalar", VarAccessMode.READ_ONLY)],
        synapse_dynamics_code=
        """
        addToPre(g * x);
        """)

    current_source_model = create_current_source_model(
        "current_source",
        vars=[("magnitude", "scalar", VarAccess.READ_ONLY)],
        injection_code=
        """
        injectCurrent(magnitude);
        """)

    model = make_model(precision, "test_additional_input_vars", backend=backend)
    model.dt = 1.0

    # Create source and target populations with different values of x
    source_x = np.arange(10.0)
    target_x = np.arange(100.0, 200.0, 10.0)
    var_init = {"isyn": 0.0, "isynA": 0.0, "isynB": 0.0}
    source_pop = model.add_neuron_population(
        "Source", 10, neuron_model,
        {}, {"x": source_x, **var_init})
    target_pop = model.add_neuron_population(
        "Target", 10, neuron_model,
        {}, {"x": target_x, **var_init})

    # Add synapse population targetting default input variable
    model.add_synapse_population(
        "DefaultSynapse", "SPARSE",
        source_pop, target_pop,
        init_weight_update(continuous_weight_update_model, {"g": 1.0},
                           pre_var_refs={"x": create_var_ref(source_pop, "x")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))

    # Add two synapse populations targetting first additional input variable
    additional_s_pop_1 = model.add_synapse_population(
        "AdditionalSynapse1", "SPARSE",
        source_pop, target_pop,
        init_weight_update(continuous_weight_update_model, {"g": 2.0},
                           pre_var_refs={"x": create_var_ref(source_pop, "x")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    additional_s_pop_1.post_target_var = "IsynA"

    additional_s_pop_2 = model.add_synapse_population(
        "AdditionalSynapse2", "SPARSE",
        source_pop, target_pop,
        init_weight_update(continuous_weight_update_model, {"g": 3.0},
                           pre_var_refs={"x": create_var_ref(source_pop, "x")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    additional_s_pop_2.post_target_var = "IsynA"

    # Add current source targetting second additional input variable
    magnitude = np.arange(0.0, 1000.0, 100.0)
    cs = model.add_current_source("CurrentSource", current_source_model,
                                  target_pop, {}, {"magnitude": magnitude})
    cs.target_var = "IsynB"

    # Add synapse population providing presynaptic
    # input to second additional input variable
    reverse_s_pop = model.add_synapse_population(
        "ReverseSynapse", "SPARSE",
        source_pop, target_pop,
        init_weight_update(continuous_reverse_weight_update_model, {"g": 4.0},
                           post_var_refs={"x": create_var_ref(target_pop, "x")}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    reverse_s_pop.pre_target_var = "IsynB"

    # Build model and load
    model.build()
    model.load()

    # Additional input variables should be reset to their initial
    # value every timestep, before each source of input is added
    correct = [(source_pop, "isyn", np.zeros(10)),
               (source_pop, "isynA", np.full(10, ISYN_A_INIT)),
               (source_pop, "isynB", ISYN_B_INIT + (4.0 * target_x)),
               (target_pop, "isyn", source_x),
               (target_pop, "isynA", ISYN_A_INIT + (5.0 * source_x)),
               (target_pop, "isynB", ISYN_B_INIT + magnitude)]
    while model.timestep < 10:
        model.step_time()

        # Loop through populations and variables
        for pop, var_name, correct_values in correct:
            pop.vars[var_name].pull_from_device()
            values = pop.vars[var_name].values

            if not np.allclose(values, correct_values):
                assert False, f"{pop.name} var {var_name} has wrong value ({values} rather than {correct_values})"
