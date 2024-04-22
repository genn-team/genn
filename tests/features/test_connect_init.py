import numpy as np
import pytest
from pygenn import types
from scipy import stats

from pygenn import (create_neuron_model, init_postsynaptic, 
                    init_sparse_connectivity, init_weight_update)

# Neuron model which does nothing
empty_neuron_model = create_neuron_model("empty")

@pytest.mark.flaky
@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_connect_init(make_model, backend, precision):
    model = make_model(precision, "test_connect_init", backend=backend)
    model.narrow_sparse_ind_enabled = True
    
    # Create pre and postsynaptic neuron populations
    pre_pop = model.add_neuron_population("Pre", 100, empty_neuron_model)
    post_pop = model.add_neuron_population("Post", 100, empty_neuron_model)
    
    # Add synapse populations with different types of built-in connectivity
    fixed_number_total_s_pop = model.add_synapse_population(
        "FixedNumberTotal", "SPARSE",
        pre_pop, post_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("FixedNumberTotalWithReplacement", {"num": 1000}))

    fixed_number_pre_s_pop = model.add_synapse_population(
        "FixedNumberPre", "SPARSE",
        pre_pop, post_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": 10}))
    
    fixed_number_post_s_pop = model.add_synapse_population(
        "FixedNumberPost", "SPARSE",
        pre_pop, post_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("FixedNumberPostWithReplacement", {"num": 10}))

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
    
    """
    After considerable thought as to why these fail:
     * Each connectivity is tested in single and double precisison
     * Each connectivity is tested using 5 different RNGs (OpenCL, CUDA, MSVC standard library, Clang standard library, GCC standard library)
     = 10 permutations
     We want the probability that one or more of the 5 tests fail simply by chance 
     to be less than 2%; for significance level a the probability that none of the 
     tests fail is (1-a)^60 which we want to be 0.98, i.e. 98% of the time the test 
    passes if the algorithm is correct. Hence, a= 1- 0.98^(1/10) = 0.0020
    """
    # Check neurons are uniformly distributed within each row/column
    confidence_interval = 0.0020
    assert stats.chisquare(np.bincount(fixed_number_post_s_pop.get_sparse_post_inds(), minlength=100)).pvalue > confidence_interval
    assert stats.chisquare(np.bincount(fixed_number_pre_s_pop.get_sparse_pre_inds(), minlength=100)).pvalue > confidence_interval
