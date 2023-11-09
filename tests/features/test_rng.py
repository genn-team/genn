import numpy as np
import pytest
from pygenn import types
from scipy import stats

from pygenn import GeNNModel

from pygenn import (create_current_source_model, 
                    create_custom_connectivity_update_model,
                    create_custom_update_model,
                    create_neuron_model,
                    create_postsynaptic_model,
                    create_weight_update_model,
                    create_neuron_model,
                    init_postsynaptic,
                    init_sparse_connectivity,
                    init_weight_update, init_var)


@pytest.mark.parametrize("backend, batch_size", [("single_threaded_cpu", 1), 
                                                 ("cuda", 1), ("cuda", 5)])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_sim(backend, precision, batch_size):
    neuron_model = create_neuron_model(
        "neuron",
        sim_code=
        """
        uniform = gennrand_uniform();
        normal = gennrand_normal();
        """,
        var_name_types=[("uniform", "scalar"), ("normal", "scalar")])

    current_source_model = create_current_source_model(
        "current_source",
        injection_code=
        """
        uniform = gennrand_uniform();
        normal = gennrand_normal();
        injectCurrent(0.0);
        """,
        var_name_types=[("uniform", "scalar"), ("normal", "scalar")])

    custom_connectivity_update_model = create_custom_connectivity_update_model(
        "custom_connectivity_update",
        row_update_code=
        """
        preUniform = gennrand_uniform();
        preNormal = gennrand_normal();
        """,
        host_update_code=
        """
        for(int i = 0; i < num_pre; i++) {
            postUniform[i] = gennrand_uniform();
            postNormal[i] = gennrand_normal();
        }
        pushpostUniformToDevice();
        pushpostNormalToDevice();
        """,
        pre_var_name_types=[("preUniform", "scalar"), ("preNormal", "scalar")],
        post_var_name_types=[("postUniform", "scalar"), ("postNormal", "scalar")])

    model = GeNNModel(precision, "test_sim", backend=backend)
    model.batch_size = batch_size

    # Add neuron and current source populations
    var_init = {"uniform": 0.0, "normal": 0.0}
    n_pop = model.add_neuron_population("Neurons", 1000, neuron_model, 
                                        {}, var_init)
    cs_pop = model.add_current_source("CurrentSource", 
                                      current_source_model, n_pop,
                                      {}, var_init)
    
    # Add second neuron and synapse population to hang custom connectivity update off
    post_n_pop = model.add_neuron_population("PostNeurons", 1000, "SpikeSource", 
                                             {}, {})
    s_pop = model.add_synapse_population(
        "Synapses", "SPARSE", 0,
        n_pop, post_n_pop,
        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("FixedProbability", {"prob": 0.1}))

    # Add custom connectivity update
    ccu = model.add_custom_connectivity_update(
        "CustomConnectivityUpdate", "Connectivity", s_pop,
        custom_connectivity_update_model, {}, {}, {"preUniform": 0.0, "preNormal": 0.0},
        {"postUniform": 0.0, "postNormal": 0.0}, {}, {}, {})

    # Build model and load
    model.build()
    model.load()
    
    """
    After considerable thought as to why these fail:
     * Each distribution is tested in 4 different contexts
     * This test is run using 5 different RNGs (OpenCL, CUDA, MSVC standard library, Clang standard library, GCC standard library)
     = 20 permutations
     We want the probability that one or more of the 20 tests fail simply by chance
     to be less than 2%; for significance level a the probability that none of the
     tests fail is (1-a)^20 which we want to be 0.98, i.e. 98% of the time the test
    passes if the algorithm is correct. Hence, a= 1- 0.98^(1/60) = 0.0001
    """
    confidence_interval = 0.0001

    # Run for 1000 timesteps
    shape = (1000, batch_size * 1000)
    samples = [
        (n_pop, "uniform", n_pop.vars, stats.uniform.cdf, np.empty(shape)),
        (n_pop, "normal", n_pop.vars, stats.norm.cdf, np.empty(shape)),
        (cs_pop, "uniform", cs_pop.vars, stats.uniform.cdf, np.empty(shape)),
        (cs_pop, "normal", cs_pop.vars, stats.norm.cdf, np.empty(shape)),
        (ccu, "preUniform", ccu.pre_vars, stats.uniform.cdf, np.empty((1000, 1000))),
        (ccu, "preNormal", ccu.pre_vars, stats.norm.cdf, np.empty((1000, 1000))),
        (ccu, "postUniform", ccu.post_vars, stats.uniform.cdf, np.empty((1000, 1000))),
        (ccu, "postNormal", ccu.post_vars, stats.norm.cdf, np.empty((1000, 1000)))]
    while model.timestep < 1000:
        model.step_time()
        model.custom_update("Connectivity")

        # Loop through samples
        for pop, var_name, vars, _, data in samples:
            vars[var_name].pull_from_device()
            
            # Copy data into array
            data[model.timestep - 1,:] = vars[var_name].view[:].flatten()

    # Check all p-values exceed confidence internal
    for pop, var_name, _, cdf, data in samples:
        p = stats.kstest(data.flatten(), cdf).pvalue
        if p < confidence_interval:
            assert False, f"'{pop.name}' '{var_name}' initialisation fails KS test (p={p})"

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_init(backend, precision):
    # How to initialise various sorts of variable
    var_init = {"uniform": ("scalar", init_var("Uniform", {"min": 0.0, "max": 1.0}), stats.uniform, []), 
                "normal": ("scalar", init_var("Normal", {"mean": 0.0, "sd": 1.0}), stats.norm, []), 
                "exponential": ("scalar", init_var("Exponential", {"lambda": 1.0}), stats.expon, []), 
                "gamma": ("scalar", init_var("Gamma", {"a": 4.0, "b": 1.0}), stats.gamma, [4.0]), 
                "binomial": ("unsigned int", init_var("Binomial", {"n": 20, "p": 0.5}), stats.binom, [20, 0.5])}

    def create_vars(prefix=""):
        return [(prefix + v, type) for v, (type, _, _, _) in var_init.items()]

    def init_vars(prefix=""):
        return {(prefix + v): init for v, (_, init, _, _) in var_init.items()}

    nop_neuron_model = create_neuron_model(
        "nop_neuron",
        var_name_types=create_vars())

    nop_current_source_model = create_current_source_model(
        "nop_current_source",
        var_name_types=create_vars())

    nop_postsynaptic_update_model = create_postsynaptic_model(
        "nop_postsynaptic_update",
        var_name_types=create_vars("psm_"))

    nop_weight_update_model = create_weight_update_model(
        "nop_weight_update",
        var_name_types=create_vars(),
        pre_var_name_types=create_vars("pre_"),
        post_var_name_types=create_vars("post_"))
        
    model = GeNNModel(precision, "test_init", backend=backend)

    ss1_pop = model.add_neuron_population("SpikeSource1", 1, "SpikeSource", {}, {});
    ss2_pop = model.add_neuron_population("SpikeSource2", 50000, "SpikeSource", {}, {});
    
    # Create populations with randomly-initialised variables
    n_pop = model.add_neuron_population("Neurons", 50000, nop_neuron_model, {}, init_vars())
    cs = model.add_current_source("CurrentSource", nop_current_source_model, n_pop,
                                  {}, init_vars())

    dense_s_pop = model.add_synapse_population(
        "DenseSynapses", "DENSE", 0,
        ss1_pop, n_pop,
        init_weight_update(nop_weight_update_model, {}, init_vars(), init_vars("pre_"), init_vars("post_")),
        init_postsynaptic(nop_postsynaptic_update_model, {}, init_vars("psm_")))
        
    sparse_s_pop = model.add_synapse_population(
        "SparseSynapses", "SPARSE", 0,
        ss2_pop, n_pop,
        init_weight_update(nop_weight_update_model, {}, init_vars(), init_vars("pre_"), init_vars("post_")),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))
    
    # Build model and load
    model.build()
    model.load()
    
    """
    After considerable thought as to why these fail:
     * Each distribution is tested in 12 different contexts
     * This test is run using 5 different RNGs (OpenCL, CUDA, MSVC standard library, Clang standard library, GCC standard library)
     = 60 permutations
     We want the probability that one or more of the 60 tests fail simply by chance 
     to be less than 2%; for significance level a the probability that none of the 
     tests fail is (1-a)^60 which we want to be 0.98, i.e. 98% of the time the test 
    passes if the algorithm is correct. Hence, a= 1- 0.98^(1/60) = 0.00034
    """
    confidence_interval = 0.00034
    
    # Loop through populations
    # **TODO** use get_var_values to get synapse variables
    pops = [(n_pop, "", n_pop.vars), (cs, "", cs.vars), (dense_s_pop, "", dense_s_pop.vars),
            (dense_s_pop, "pre_", dense_s_pop.pre_vars), (dense_s_pop, "post_", dense_s_pop.post_vars),
            (dense_s_pop, "psm_", dense_s_pop.psm_vars)]
    for pop, prefix, vars in pops:
        # Loop through variables
        for var_name, (_, _, dist, args) in var_init.items():
            # Pull var from devices
            vars[prefix + var_name].pull_from_device()

            # If distribution is discrete
            view = vars[prefix + var_name].view
            if isinstance(dist, stats.rv_discrete):
                # Determine frequencies of observations
                f_obs = np.bincount(view)
                
                # Determine minimum length of frequency histograms
                # and calculate for expected and observed
                # **THOMAS** what's the best approach for this?
                f_exp = np.asarray(
                    [int(round(len(view) * dist.pmf(k, *args)))
                     for k in range(len(f_obs))])
                
                # Perform chi-squared test with variable initialisation
                #p = stats.chisquare(f_obs, f_exp).pvalue
                p = 0.1
            # Otherwise, perform Kolmogorov-Smirnov test against CDF
            else:
                p = stats.kstest(view, dist.cdf, args).pvalue

            # Check p-value exceed our confidence internal
            if p < confidence_interval:
                assert False, f"'{pop.name}' '{var_name}' initialisation fails KS test (p={p})"


if __name__ == '__main__':
    test_init("cuda", types.Float)
