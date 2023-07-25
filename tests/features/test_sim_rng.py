import numpy as np
import pytest
from pygenn import types
from scipy import stats

from pygenn import GeNNModel

from pygenn import (create_current_source_model, 
                    create_custom_connectivity_update_model,
                    create_neuron_model,
                    init_sparse_connectivity)

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

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_sim_rng(backend, precision):
    model = GeNNModel(precision, "test_sim_rng", backend=backend)
    
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
    s_pop = model.add_synapse_population("Synapses", "SPARSE", 0,
                                         n_pop, post_n_pop,
                                         "StaticPulseConstantWeight", {"g": 1.0}, {}, {}, {},
                                         "DeltaCurr", {}, {},
                                         init_sparse_connectivity("FixedProbability", {"prob": 0.1}))

    # Add custom connectivity update
    ccu = model.add_custom_connectivity_update(
        "CustomConnectivityUpdate", "Connectivity", s_pop,
        custom_connectivity_update_model, {}, {}, {"preUniform": 0.0, "preNormal": 0.0},
        {"postUniform": 0.0, "postNormal": 0.0}, {}, {}, {})

    # Build model and load
    model.build()
    model.load()
    
    # Run for 1000 timesteps
    samples = [
        (n_pop, "uniform", n_pop.vars, stats.uniform.cdf, np.empty((1000, 1000))),
        (n_pop, "normal", n_pop.vars, stats.norm.cdf, np.empty((1000, 1000))),
        (cs_pop, "uniform", cs_pop.vars, stats.uniform.cdf, np.empty((1000, 1000))),
        (cs_pop, "normal", cs_pop.vars, stats.norm.cdf, np.empty((1000, 1000))),
        (ccu, "preUniform", ccu.pre_vars, stats.uniform.cdf, np.empty((1000, 1000))),
        (ccu, "preNormal", ccu.pre_vars, stats.norm.cdf, np.empty((1000, 1000))),
        (ccu, "postUniform", ccu.post_vars, stats.uniform.cdf, np.empty((1000, 1000))),
        (ccu, "postNormal", ccu.post_vars, stats.norm.cdf, np.empty((1000, 1000)))]
    while model.timestep < 1000:
        model.step_time()
        model.custom_update("Connectivity")

        # Loop through samples
        for pop, var_name, vars, _, data in samples:
            pop.pull_var_from_device(var_name)
            
            # Copy data into array
            data[model.timestep - 1,:] = vars[var_name].view[:]
    
    # Check all p-values exceed 95% confidence internal
    for pop, var_name, _, cdf, data in samples:
        p = stats.kstest(data.flatten(), cdf).pvalue
        if p < 0.05:
            assert False, f"{pop.name} '{var_name} initialisation failes KS test (p={p})"


if __name__ == '__main__':
    test_sim_rng("single_threaded_cpu", types.Double)