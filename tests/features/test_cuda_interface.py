import numpy as np
import pytest
import sys

try:
    import cupy as cp
except ImportError:
    pytest.skip("CuPy is required for this test", allow_module_level=True)

from pygenn import genn_model

@pytest.mark.parametrize("precision,np_dtype", [("float", np.float32), ("double", np.float64)])
def test_cuda_array_interface(make_model, backend, precision, np_dtype):
    """
    Verification test for CUDA Array Interface implementation in GeNN.

    This test verifies that:
    1. GeNN arrays correctly implement the __cuda_array_interface__ property
    2. CuPy can access GeNN's device memory directly through this interface
    3. Modifications made by CuPy are reflected in GeNN's arrays
    4. The entire data exchange happens without unnecessary memory copies
    """
    if backend != "cuda":
        pytest.skip("CUDA Array Interface test requires CUDA backend")
    
    model = make_model(precision, "verify_cuda_interface", backend=backend)
    
    neurons = model.add_neuron_population(
        "neurons", 100, "LIF", 
        {
            "C": 1.0,
            "TauM": 20.0,
            "Vrest": -65.0,
            "Vreset": -70.0,
            "Vthresh": -55.0,
            "Ioffset": 0.0,
            "TauRefrac": 0.0
        }, 
        {
            "V": -65.0,
            "RefracTime": 0.0
        }
    )
    
    model.build()
    model.load()
    
    init_values = np.linspace(-70.0, -60.0, 100, dtype=np_dtype)
    neurons.vars["V"].view[:] = init_values
    neurons.vars["V"].push_to_device()
    
    v_cupy = cp.asarray(neurons.vars["V"])
    
    v_cupy += 10.0
    
    neurons.vars["V"].pull_from_device()
    modified_values = neurons.vars["V"].view
    
    expected = init_values + 10.0
    assert np.allclose(modified_values, expected), \
        f"Expected: {expected[:5]}..., Got: {modified_values[:5]}..."

if __name__ == "__main__":
    pytest.main() 
