Release Notes for GeNN v2.2
====

This release includes minor new features, some core code improvements and several bug fixes on GeNN v2.1.

User Side Changes
----

1. There is now a new mechanism for how frequently changing "external input" is communicated to kernels. Rather than a GeNN generated argument list for each kernel comprising direct inputs, extraglobal parameters and the time variable in GeNN <= 2.1, there are now no kernel arguments other than the global time t for any of the kernels and the relevant variables are copied to the GPU in a single memory copy at each time step.
For users this means that within kernels, references need to be made to the correctly named variables and similarly when setting them on the CPU side.
Examples: CPU  	     GPU (kernel code snippet)
	  t	     $(t)
	  extraHH    $(extra)
	  (for an extraglobalneuronparameter "extra" in a neurongroup "HH")

Note, that for example, the predefined Poisson neurons have an auto-copy for the pointer to the rates array and for the offset (integer) where to use the array. The rates themselves in the array however are not updated automatically (this is exactly as before with the kernel arguments).
IMPORTANT NOTE: The global time variable "t" is now provided by GeNN; please make sure that you are not duplicating its definition or shadowing it. This could have severe consequences for simulation correctness (e.g. time not advancing in cases of over-shadowing).
The concept of "directInput" has been removed. Users can easily achieve the same thing by adding an additional variable (if there are individual inputs to neurons), an extraGlobalNeuronParameter (if the input is homogeneous but time dependent) or, obviously, a simple parameter if it's homogeneous and constant.

2. We introduced the namespace GENN_PREFERENCES which contains variables that determine the behaviour of GeNN. These include

3. We introduced a new code snippet called "supportCode" for neuron models, weightupdate models and post-synaptic models. This code snippet is intended to contain user-defined functions that are used from the other code snippets. We advise where possible to define the support code functions with keywords "__host__ __device__" so that they are available for both GPU and CPU version. Alternatively one can define separate versions for __host__ and __device__ in the snippet. The snippets are automatically made available to the relevant code parts. This is regulated through namespaces so that name clashes between different models do not matter. An exception are hash defines. They can in principle be used in the supportCode snippet but need to be protected specifically using #ifndef. Example
#ifndef clip(x)
#define clip(x) x > 10.0? 10.0 : x
#endif
Note: If there are conflicting definitions for hash defines, the one that appears first in the GeNN generated code will then prevail.

4. The new convenience macros spikeCount_XX and spike_XX where "XX" is the name of the neuron group are now also available fro events: spikeEventCount_XX and spikeEvent_XX. They access the values for the current time step even if there are synaptic delays and spikes events are stored in circular queues.

5. We have now introduced a "CPU_ONLY" macro that if it's defined will generate a GeNN version that is completely independent from CUDA and hence can be used on computers without CUDA installation or CUDA enabled hardware. Obviously, this then can also only run on CPU. CPU only mode can either be switched on by defining CPU_ONLY in the model description file or by passing appropriate parameters during the build, in particular
buildmodel <name> CPU_ONLY=1
make release CPU_ONLY=1
**TODO** how to handle the problem with the main executable name suffix .cu

Developer Side Changes
----

1. Blocksize optimization and device choice now obtain the ptxas information on memory usage from a CUDA driver API call rather than from parsing ptxas output of the nvcc compiler. This adds robustness to any change in the syntax of the compiler output.

2. The information about device choice is now stored in variables in the namespace GENN_PREFERENCES. This includes `chooseDevice`, `optimiseBlockSize`, `defaultDevice`. `asGoodAsZero` has also been moved into this namespace.

3. We have also introduced the namespace GENN_FLAGS that contains unsigned int variables that attach names to mueric flags that can be used within GeNN. 

4. The headers of the auto-generated communication functions such as pullXXXStateFromDevice etc, are now generated into definitions.h. This is useful where one wants to compile separate object files that cannot all include the full definitions in "runnerGPU.cc". One example where this is useful is the brian2genn interface.


Improvements
----
1. Improved method of obtaining ptxas compiler information on register and shared memory usage and an improved algorithm for estimating shared memory usage requirements for different block sizes.

Bug fixes:
----

Please refer to the [full documentation](http://genn-team.github.io/genn/documentation/html/index.html) for further details, tutorials and complete code documentation.
