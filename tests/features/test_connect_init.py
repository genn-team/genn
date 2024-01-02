import numpy as np
import pytest
from pygenn import types
from scipy import stats

from pygenn import (init_postsynaptic, init_sparse_connectivity,
                    init_weight_update)

@pytest.mark.flaky
@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_connect_init(make_model, backend, precision):
    model = make_model(precision, "test_connect_init", backend=backend)
    model.narrow_sparse_ind_enabled = True
    
    # Create pre and postsynaptic neuron populations
    pre_pop = model.add_neuron_population("Pre", 100, "SpikeSource", {}, {})
    post_pop = model.add_neuron_population("Post", 100, "SpikeSource", {}, {})
    
    # Add synapse populations with different types of built-in connectivity
    fixed_number_total_s_pop = model.add_synapse_population(
        "FixedNumberTotal", "SPARSE",
        pre_pop, post_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("FixedNumberTotalWithReplacement", {"total": 1000}))

    fixed_number_pre_s_pop = model.add_synapse_population(
        "FixedNumberPre", "SPARSE",
        pre_pop, post_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("FixedNumberPreWithReplacement", {"colLength": 10}))
    
    fixed_number_post_s_pop = model.add_synapse_population(
        "FixedNumberPost", "SPARSE",
        pre_pop, post_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("FixedNumberPostWithReplacement", {"rowLength": 10}))

    # Build and load model
    model.build()
    model.load()
    
    # Pull connectivity
    fixed_number_total_s_pop.pull_connectivity_from_device()
    fixed_number_pre_s_pop.pull_connectivity_from_device()
    fixed_number_post_s_pop.pull_connectivity_from_device()
    
    # Check connectivity
    assert np.all(np.bincount(fixed_number_post_s_pop.get_sparse_pre_inds()) == 10)
    assert np.all(np.bincount(fixed_number_pre_s_pop.get_sparse_post_inds()) == 10)
    assert len(fixed_number_total_s_pop.get_sparse_pre_inds()) == 1000

    # Check neurons are uniformly distributed within each row/column
    assert stats.chisquare(np.bincount(fixed_number_post_s_pop.get_sparse_post_inds(), minlength=100)).pvalue > 0.05
    assert stats.chisquare(np.bincount(fixed_number_pre_s_pop.get_sparse_pre_inds(), minlength=100)).pvalue > 0.05
