Release Notes for GeNN v2.1
====

This release includes some new features and several bug fixes 
on GeNN v2.0.

User Side Changes
----

1. Block size debugging flag and the asGoodAsZero variables are moved into include/global.h.

2. NGRADSYNAPSES dynamics have changed (See Bug fix #4) and this change is applied to the example projects. If you are using this synapse model,
you may want to consider changing model parameters.

3. The delay slots are now such that NO_DELAY is 0 delay slots (previously 1) and 1 means an actual delay of 1 time step.

4. The convenience function convertProbabilityToRandomNumberThreshold(float *, uint64_t *, int) was changed so that it actually converts firing probability/timestep into a threshold value for the GeNN random number generator (as its name always suggested). The previous functionality of converting a *rate* in kHz into a firing threshold number for the GeNN random number generator is now provided with the name convertRateToRandomNumberThreshold(float *, uint64_t *, int)

5. Every model definition function `modelDefinition()` now needs to end with calling `NNmodel::finalize()` for the defined network model. This will lock down the model and prevent any further changes to it by the supported methods. It also triggers necessary analysis of the model structure that should only be performed once. If the `finalize()` function is not called, GeNN will issue an error and exit before code generation.

6. To be more consistent in function naming the `pull<SYNAPSENAME>FromDevice` and `push<SYNAPSENAME>ToDevice` have been renamed to `pull<SYNAPSENAME>StateFromDevice` and `push<SYNAPSENAME>StateToDevice`. The old versions are still supported through macro definitions to make the transition easier.

7. New convenience macros are now provided to access the current spike numbers and identities of neurons that spiked. These are called spikeCount_XX and spike_XX where "XX" is the name of the neuron group. They access the values for the current time step even if there are synaptic delays and spikes are stored in circular queues.

8. There is now a pre-defined neuron type "SPIKECOURCE" which is empty and can be used to define PyNN style spike source arrays. 

9. The macros FLOAT and DOUBLE were replaced with GENN_FLOAT and GENN_DOUBLE due to name clashes with typedefs in Windows that define FLOAT and DOUBLE.

Developer Side Changes
----

1. We introduced a file definitions.h, which is generated and filled with useful macros such as spkQuePtrShift which tells users where in the circular spike queue their spikes start.

Improvements
----

1. Improved debugging information for block size optimisation
and device choice.

2. Changed the device selection logic so that device occupancy has larger priority than device capability version.

3. A new HH model called TRAUBMILES\_PSTEP where one can set the number of inner loops as a parameter is introduced. It uses the TRAUBMILES\_SAFE method. 

4. An alternative method is added for the insect olfaction model in order
to fix the number of connections to a maximum of 10K in order to avoid
negative conductance tails.

5. We introduced a #define for an "int_" function that translates floating points to integers.

Bug fixes:
----

1. AtomicAdd replacement for old GPUs were used by mistake if the model runs in double precision. 

2. Timing of individual kernels is fixed and improved.

3. More careful setting of maximum number of connections in sparse connectivity, covering mixed dense/sparse network scenarios.

4. NGRADSYNAPSES was not scaling correctly with varying time step. 

5. Fixed a bug where learning kernel with sparse connectivity was going out of range in an array.

6. Fixed synapse kernel name substitutions where the "dd_" prefix was omitted by mistake.

Please refer to the [full documentation](http://genn-team.github.io/genn/documentation/html/index.html) for further details, tutorials and complete code documentation.
