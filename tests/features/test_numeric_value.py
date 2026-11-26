import numpy as np
import pytest
from pygenn import types

from pygenn import (create_neuron_model, create_sparse_connect_init_snippet,
                    init_postsynaptic, init_sparse_connectivity,
                    init_weight_update)
from pygenn._genn import NumericValue

# Neuron model which does nothing
empty_neuron_model = create_neuron_model("empty")

@pytest.mark.parametrize("value, value_type",
                         [(3, int), (-7, int), (True, int), (3.5, float),
                          (np.int32(3), int), (np.int64(3), int),
                          (np.uint8(3), int), (np.array(3), int),
                          (np.float64(3.5), float), (np.float32(3.5), float),
                          (np.float16(3.5), float), (np.array(3.5), float)])
def test_numeric_value(value, value_type):
    # Check that integers stay integers and that
    # floating point values don't get truncated
    numeric_value = NumericValue(value).value
    assert type(numeric_value) is value_type
    assert numeric_value == value

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_integer_param_max_row_length(make_model, backend, precision):
    # Sparse connectivity initialisation snippet whose maximum
    # row length is calculated directly from an integer parameter
    fixed_number_post_model = create_sparse_connect_init_snippet(
        "fixed_number_post",
        params=[("num", "unsigned int")],
        row_build_code=
        """
        for(unsigned int i = 0; i < num; i++) {
            addSynapse(i);
        }
        """,
        calc_max_row_len_func=lambda num_pre, num_post, pars: pars["num"])

    model = make_model(precision, "test_integer_param_max_row_length",
                       backend=backend)

    # Create pre and postsynaptic neuron populations
    pre_pop = model.add_neuron_population("Pre", 4, empty_neuron_model)
    post_pop = model.add_neuron_population("Post", 4, empty_neuron_model)

    # Add synapse population
    s_pop = model.add_synapse_population(
        "Syn", "SPARSE",
        pre_pop, post_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity(fixed_number_post_model, {"num": 3}))

    assert s_pop.max_connections == 3

    # Build model and load
    model.build()
    model.load()

    # Pull connectivity and check each presynaptic neuron
    # is connected to the first three postsynaptic neurons
    s_pop.pull_connectivity_from_device()
    assert np.array_equal(s_pop.get_sparse_pre_inds(), np.repeat(np.arange(4), 3))
    assert np.array_equal(s_pop.get_sparse_post_inds(), np.tile(np.arange(3), 4))
