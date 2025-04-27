#!/usr/bin/env python3

"""
Verification script for CUDA Array Interface implementation in GeNN.

This test verifies that:
1. GeNN arrays correctly implement the __cuda_array_interface__ property
2. CuPy can access GeNN's device memory directly through this interface
3. Modifications made by CuPy are reflected in GeNN's arrays
4. The entire data exchange happens without unnecessary memory copies

This interoperability feature enables GeNN to be used in workflows with
other Python libraries like CuPy, PyTorch, and other frameworks that
implement the CUDA Array Interface protocol.
"""

import numpy as np
import sys

try:
    import cupy as cp
except ImportError:
    print("CuPy is required for this verification")
    sys.exit(1)

from pygenn import genn_model

def verify_implementation():
    model = genn_model.GeNNModel("float", "verify_cuda_interface", "cuda")
    
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
    
    init_values = np.linspace(-70.0, -60.0, 100)
    neurons.vars["V"].view[:] = init_values
    neurons.vars["V"].push_to_device()
    
    v_cupy = cp.asarray(neurons.vars["V"])
    
    v_cupy += 10.0
    
    neurons.vars["V"].pull_from_device()
    modified_values = neurons.vars["V"].view
    
    expected = init_values + 10.0
    is_correct = np.allclose(modified_values, expected)
    
    if is_correct:
        print("CUDA Array Interface verification: SUCCESS")
        print("CuPy was able to modify GeNN memory directly")
        return True
    else:
        print("CUDA Array Interface verification: FAILED")
        print(f"Expected: {expected[:5]}...")
        print(f"Got: {modified_values[:5]}...")
        return False

if __name__ == "__main__":
    success = verify_implementation()
    sys.exit(0 if success else 1) 
