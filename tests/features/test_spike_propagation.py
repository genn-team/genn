import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel

from pygenn.genn import VarAccess
from pygenn import (create_neuron_model,
                    create_sparse_connect_init_snippet,
                    init_sparse_connectivity)

decoder_model = create_sparse_connect_init_snippet(
    "decoder",
    row_build_code=
    """
    for(unsigned int j = 0; j < num_post; j++) {
       const unsigned int jValue = (1 << j);
       if(((id_pre + 1) & jValue) != 0) {
           addSynapse(j);
       }
    }
    """)

pre_cont_neuron_model = create_neuron_model(
    "pre_cont_neuron",
    var_name_types=[("x", "scalar", VarAccess.READ_ONLY)])

post_neuron_model = create_neuron_model(
    "post_neuron",
    sim_code=
    """
    x = Isyn;
    """,
    var_name_types=[("x", "scalar")])

# decode_matrix_conn_gen_globalg_ragged, decode_matrix_conn_gen_globalg_bitmask,
# decode_matrix_conn_gen_globalg_bitmask_optimised, decode_matrix_conn_gen_individualg_ragged
@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_spike_propagation_snippet(backend, precision):
    model = GeNNModel(precision, "test_spike_propagation", backend=backend)
    model.dt = 1.0

    # Create spike source array to generate one-hot pattern to decode
    ss_pop = model.add_neuron_population("SpikeSource", 16, "SpikeSourceArray",
                                         {}, {"startSpike": np.arange(16), "endSpike": np.arange(1, 17)})
    ss_pop.extra_global_params["spikeTimes"].set_values(np.arange(16.0))

    # Create one output neuron pop with constant weight sparse decoder population
    sparse_constant_weight_n_pop = model.add_neuron_population(
        "PostSparseConstantWeightNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "SparseConstantWeightSynapse", "SPARSE", 0,
        ss_pop, sparse_constant_weight_n_pop,
        "StaticPulseConstantWeight", {"g": 1.0}, {}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity(decoder_model, {}))

    # Create one output neuron pop with sparse decoder population
    sparse_n_pop = model.add_neuron_population(
        "PostSparseNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "SparseSynapse", "SPARSE", 0,
        ss_pop, sparse_n_pop,
        "StaticPulse", {}, {"g": 1.0}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity(decoder_model, {}))
    
    # Create one output neuron pop with bitmask decoder population
    bitmask_n_pop = model.add_neuron_population(
        "PostBitmaskNeuron", 4, post_neuron_model, 
        {}, {"x": 0.0})
    model.add_synapse_population(
        "BitmaskSynapse", "SPARSE", 0,
        ss_pop, bitmask_n_pop,
        "StaticPulseConstantWeight", {"g": 1.0}, {}, {}, {},
        "DeltaCurr", {}, {},
        init_sparse_connectivity(decoder_model, {}))
    
    # Build model and load
    model.build()
    model.load()

    # Simulate 16 timesteps
    output_place_values = 2 ** np.arange(4)
    output_populations = [sparse_constant_weight_n_pop, 
                          sparse_n_pop, bitmask_n_pop]
    while model.timestep < 16:
        model.step_time()
        
        # Loop through output populations
        for pop in output_populations:
            # Pull state variable
            pop.pull_var_from_device("x")

            # Convert to binary mask
            output_binary = np.isclose(np.ones(4), pop.vars["x"].view)
            
            # Sum up active place values
            output_value = np.sum(output_place_values[output_binary])
            if output_value != (model.timestep - 1):
                assert False, f"{pop.name} decoding incorrect ({output_value} rather than {model.timestep - 1})"
    print("BEEP")

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_cont_propagation_snippet(backend, precision):
    model = GeNNModel(precision, "test_cont_propagation", backend=backend)
    

if __name__ == '__main__':
    test_spike_propagation_snippet("single_threaded_cpu", types.Float)