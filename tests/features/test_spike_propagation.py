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
    x= Isyn;
    """,
    var_name_types=[("x", "scalar")])

@pytest.mark.parametrize("backend", ["single_threaded_cpu", "cuda"])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_spike_propagation(backend, precision):
    model = GeNNModel(precision, "test_spike_propagation", backend=backend)

if __name__ == '__main__':
    test_spike_propagation("single_threaded_cpu", types.Float)