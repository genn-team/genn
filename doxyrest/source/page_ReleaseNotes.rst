.. index:: pair: page; Release Notes
.. _doxid-df/ddb/ReleaseNotes:

Release Notes
=============

Release Notes for GeNN v4.0.2
=============================

This release fixes several small issues with the generation of binary wheels for Python:

Bug fixes:
~~~~~~~~~~

#. There was a conflict between the versions of numpy used to build the wheels and the version required for the PyGeNN packages

#. Wheels were renamed to include the CUDA version which broke them.

Release Notes for GeNN v4.0.1
=============================

This release fixes several small bugs found in GeNN 4.0.0 and implements some small features:

User Side Changes
~~~~~~~~~~~~~~~~~

#. Improved detection and handling of errors when specifying model parameters and values in PyGeNN.

#. SpineML simulator is now implemented as a library which can be used directly from user applications as well as from command line tool.

Bug fixes:
~~~~~~~~~~

#. Fixed typo in ``GeNNModel.push_var_to_device`` function in PyGeNN.

#. Fixed broken support for Visual C++ 2013.

#. Fixed zero-copy mode.

#. Fixed typo in tutorial 2.

Release Notes for GeNN v4.0.0
=============================

This release is the result of a second round of fairly major refactoring which we hope will make GeNN easier to use and allow it to be extended more easily in future. However, especially if you have been using GeNN 2.XX syntax, it breaks backward compatibility.

User Side Changes
~~~~~~~~~~~~~~~~~

#. Totally new build system - ``make install`` can be used to install GeNN to a system location on Linux and Mac and Windows projects work much better in the Visual Studio IDE.

#. Python interface now supports Windows and can be installed using binary 'wheels' (see :ref:`Python interface (PyGeNN) <doxid-d0/d81/PyGeNN>` for more details).

#. No need to call ``initGeNN()`` at start and ``model.finalize()`` at end of all models.

#. Initialisation system simplified - if you specify a value or initialiser for a variable or sparse connectivity, it will be initialised by your chosen backend. If you mark it as uninitialised, it is up to you to initialize it in user code between the calls to ``initialize()`` and ``initializeSparse()`` (where it will be copied to device).

#. ``genn-create-user-project`` helper scripts to create Makefiles or MSBuild projects for building user code

#. State variables can now be pushed and pulled individually using the ``pull<var name><neuron or synapse name>FromDevice()`` and ``push<var name><neuron or synapse name>ToDevice()`` functions.

#. Management of extra global parameter arrays has been somewhat automated (see :ref:`Extra Global Parameters <doxid-d0/da6/UserGuide_1extraGlobalParamSim>` for more details).

#. ``GENN_PREFERENCES`` is no longer a namespace - it's a global struct so members need to be accessed with . rather than ::.

#. ``:ref:`NeuronGroup <doxid-d7/d3b/classNeuronGroup>```, ``:ref:`SynapseGroup <doxid-dc/dfa/classSynapseGroup>```, ``:ref:`CurrentSource <doxid-d1/d48/classCurrentSource>``` and ``NNmodel`` all previously exposed a lot of methods that the user wasn't *supposed* to call but could. These have now all been made protected and are exposed to GeNN internals using derived classes (``:ref:`NeuronGroupInternal <doxid-dc/da3/classNeuronGroupInternal>```, ``:ref:`SynapseGroupInternal <doxid-dd/d48/classSynapseGroupInternal>```, ``:ref:`CurrentSourceInternal <doxid-d6/de6/classCurrentSourceInternal>```, ``:ref:`ModelSpecInternal <doxid-dc/dfa/classModelSpecInternal>```) that make them public using ``using`` directives.

#. Auto-refractory behaviour was controlled using ``GENN_PREFERENCES::autoRefractory``, this is now controlled on a per-neuron-model basis using the ``SET_NEEDS_AUTO_REFRACTORY`` macro.

#. The functions used for pushing and pulling have been unified somewhat this means that ``copyStateToDevice`` and ``copyStateFromDevice`` functions no longer copy spikes and ``pus<neuron or synapse name>SpikesToDevice`` and ``pull<neuron or synapse name>SpikesFromDevice`` no longer copy spike times or spike-like events.

#. Standard models of leaky-integrate-and-fire neuron (``:ref:`NeuronModels::LIF <doxid-d0/d6d/classNeuronModels_1_1LIF>```) and of exponentially shaped postsynaptic current (``:ref:`PostsynapticModels::ExpCurr <doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr>```) have been added.

#. When a model is built using the CUDA backend, the device it was built for is stored using it's PCI bus ID so it will always use the same device.

Deprecations
~~~~~~~~~~~~

#. Yale-format sparse matrices are no longer supported.

#. GeNN 2.X syntax for implementing neuron and synapse models is no longer supported.

#. $(addtoinSyn) = X; $(updatelinsyn); idiom in weight update models has been replaced by function style ``$(addToInSyn, X);``.

Release Notes for GeNN v3.3.0
=============================

This release is intended as the last service release for GeNN 3.X.X. Fixes for serious bugs **may** be backported if requested but, otherwise, development will be switching to GeNN 4.

User Side Changes
~~~~~~~~~~~~~~~~~

#. Postsynaptic models can now have Extra Global Parameters.

#. Gamma distribution can now be sampled using ``$(gennrand_gamma, a)``. This can be used to initialise variables using ``:ref:`InitVarSnippet::Gamma <doxid-d0/d54/classInitVarSnippet_1_1Gamma>```.

#. Experimental Python interface - All features of GeNN are now exposed to Python through the ``:ref:`pygenn <doxid-da/d6d/namespacepygenn>``` module (see :ref:`Python interface (PyGeNN) <doxid-d0/d81/PyGeNN>` for more details).

Bug fixes:
~~~~~~~~~~

#. Devices with Streaming Multiprocessor version 2.1 (compute capability 2.0) now work correctly in Windows.

#. Seeding of on-device RNGs now works correctly.

#. Improvements to accuracy of memory usage estimates provided by code generator.

Release Notes for GeNN v3.2.0
=============================

This release extends the initialisation system introduced in 3.1.0 to support the initialisation of sparse synaptic connectivity, adds support for networks with more sophisticated models of synaptic plasticity and delay as well as including several other small features, optimisations and bug fixes for certain system configurations. This release supports GCC >= 4.9.1 on Linux, Visual Studio >= 2013 on Windows and recent versions of Clang on Mac OS X.

User Side Changes
~~~~~~~~~~~~~~~~~

#. Sparse synaptic connectivity can now be initialised using small *snippets* of code run either on GPU or CPU. This can save significant amounts of initialisation time for large models. See :ref:`Sparse connectivity initialisation <doxid-d5/dd4/sectSparseConnectivityInitialisation>` for more details.

#. New 'ragged matrix' data structure for representing sparse synaptic connections supports initialisation using new sparse synaptic connecivity initialisation system and enables future optimisations. See :ref:`Synaptic matrix types <doxid-d5/d39/subsect34>` for more details.

#. Added support for pre and postsynaptic state variables for weight update models to allow more efficient implementatation of trace based STDP rules. See :ref:`Defining a new weight update model <doxid-d5/d24/sectSynapseModels_1sect34>` for more details.

#. Added support for devices with Compute Capability 7.0 (Volta) to block-size optimizer.

#. Added support for a new class of 'current source' model which allows non-synaptic input to be efficiently injected into neurons. See :ref:`Current source models <doxid-d0/d1e/sectCurrentSourceModels>` for more details.

#. Added support for heterogeneous dendritic delays. See :ref:`Defining a new weight update model <doxid-d5/d24/sectSynapseModels_1sect34>` for more details.

#. Added support for (homogeneous) synaptic back propagation delays using ``:ref:`SynapseGroup::setBackPropDelaySteps <doxid-dc/dfa/classSynapseGroup_1ac080d0115f8d3aa274e9f95898b1a443>```.

#. For long simulations, using single precision to represent simulation time does not work well. Added ``:ref:`NNmodel::setTimePrecision <doxid-da/dfd/classModelSpec_1a379793c6fcbe1f834ad18cf4c5789537>``` to allow data type used to represent time to be set independently.

Optimisations
~~~~~~~~~~~~~

#. ``GENN_PREFERENCES::mergePostsynapticModels`` flag can be used to enable the merging together of postsynaptic models from a neuron population's incoming synapse populations - improves performance and saves memory.

#. On devices with compute capability > 3.5 GeNN now uses the read only cache to improve performance of postsynaptic learning kernel.

Bug fixes:
~~~~~~~~~~

#. Fixed bug enabling support for CUDA 9.1 and 9.2 on Windows.

#. Fixed bug in SynDelay example where membrane voltage went to NaN.

#. Fixed bug in code generation of ``SCALAR_MIN`` and ``SCALAR_MAX`` values.

#. Fixed bug in substitution of trancendental functions with single-precision variants.

#. Fixed various issues involving using spike times with delayed synapse projections.

Release Notes for GeNN v3.1.1
=============================

This release fixes several small bugs found in GeNN 3.1.0 and implements some small features:

User Side Changes
~~~~~~~~~~~~~~~~~

#. Added new synapse matrix types ``SPARSE_GLOBALG_INDIVIDUAL_PSM``, ``DENSE_GLOBALG_INDIVIDUAL_PSM`` and ``BITMASK_GLOBALG_INDIVIDUAL_PSM`` to handle case where synapses with no individual state have a postsynaptic model with state variables e.g. an alpha synapse. See :ref:`Synaptic matrix types <doxid-d5/d39/subsect34>` for more details.

Bug fixes
~~~~~~~~~

#. Correctly handle aliases which refer to other aliases in SpineML models.

#. Fixed issues with presynaptically parallelised synapse populations where the postsynaptic population is small enough for input to be accumulated in shared memory.

Release Notes for GeNN v3.1.0
=============================

This release builds on the changes made in 3.0.0 to further streamline the process of building models with GeNN and includes several bug fixes for certain system configurations.

User Side Changes
~~~~~~~~~~~~~~~~~

#. Support for simulating models described using the `SpineML <http://spineml.github.io/>`__ model description language with GeNN (see :ref:`SpineML and SpineCreator <doxid-d2/dba/SpineML>` for more details).

#. Neuron models can now sample from uniform, normal, exponential or log-normal distributions - these calls are translated to `cuRAND <http://docs.nvidia.com/cuda/curand/index.html>`__ when run on GPUs and calls to the C++11 ``<random>`` library when run on CPU. See :ref:`Defining your own neuron type <doxid-de/ded/sectNeuronModels_1sect_own>` for more details.

#. Model state variables can now be initialised using small *snippets* of code run either on GPU or CPU. This can save significant amounts of initialisation time for large models. See :ref:`Defining a new variable initialisation snippet <doxid-d4/dc6/sectVariableInitialisation_1sect_new_var_init>` for more details.

#. New `MSBuild <https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-reference>`__ build system for Windows - makes developing user code from within Visual Studio much more streamlined. See :ref:`Debugging suggestions <doxid-d0/da6/UserGuide_1Debugging>` for more details.

Bug fixes:
~~~~~~~~~~

#. Workaround for `bug <https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280>`__ found in Glibc 2.23 and 2.24 which causes poor performance on some 64-bit Linux systems (namely on Ubuntu 16.04 LTS).

#. Fixed bug encountered when using extra global variables in weight updates.

Release Notes for GeNN v3.0.0
=============================

This release is the result of some fairly major refactoring of GeNN which we hope will make it more user-friendly and maintainable in the future.

User Side Changes
~~~~~~~~~~~~~~~~~

#. Entirely new syntax for defining models - hopefully terser and less error-prone (see updated documentation and examples for details).

#. Continuous integration testing using Jenkins - automated testing and code coverage calculation calculated automatically for Github pull requests etc.

#. Support for using Zero-copy memory for model variables. Especially on devices such as NVIDIA Jetson TX1 with no physical GPU memory this can significantly improve performance when recording data or injecting it to the simulation from external sensors.

Release Notes for GeNN v2.2.3
=============================

This release includes minor new features and several bug fixes for certain system configurations.

User Side Changes
~~~~~~~~~~~~~~~~~

#. Transitioned feature tests to use Google Test framework.

#. Added support for CUDA shader model 6.X

Bug fixes:
~~~~~~~~~~

#. Fixed problem using GeNN on systems running 32-bit Linux kernels on a 64-bit architecture (Nvidia Jetson modules running old software for example).

#. Fixed problem linking against CUDA on Mac OS X El Capitan due to SIP (System Integrity Protection).

#. Fixed problems with support code relating to its scope and usage in spike-like event threshold code.

#. Disabled use of C++ regular expressions on older versions of GCC.

Release Notes for GeNN v2.2.2
=============================

This release includes minor new features and several bug fixes for certain system configurations.

User Side Changes
~~~~~~~~~~~~~~~~~

#. Added support for the new version (2.0) of the Brian simulation package for Python.

#. Added a mechanism for setting user-defined flags for the C++ compiler and NVCC compiler, via ``GENN_PREFERENCES``.

Bug fixes:
~~~~~~~~~~

#. Fixed a problem with ``atomicAdd()`` redefinitions on certain CUDA runtime versions and GPU configurations.

#. Fixed an incorrect bracket placement bug in code generation for certain models.

#. Fixed an incorrect neuron group indexing bug in the learning kernel, for certain models.

#. The dry-run compile phase now stores temporary files in the current directory, rather than the temp directory, solving issues on some systems.

#. The ``LINK_FLAGS`` and ``INCLUDE_FLAGS`` in the common windows makefile include 'makefile_commin_win.mk' are now appended to, rather than being overwritten, fixing issues with custom user makefiles on Windows.

Release Notes for GeNN v2.2.1
=============================

This bugfix release fixes some critical bugs which occur on certain system configurations.

Bug fixes:
~~~~~~~~~~

#. (important) Fixed a Windows-specific bug where the CL compiler terminates, incorrectly reporting that the nested scope limit has been exceeded, when a large number of device variables need to be initialised.

#. (important) Fixed a bug where, in certain circumstances, outdated generateALL objects are used by the Makefiles, rather than being cleaned and replaced by up-to-date ones.

#. (important) Fixed an 'atomicAdd' redeclared or missing bug, which happens on certain CUDA architectures when using the newest CUDA 8.0 RC toolkit.

#. (minor) The SynDelay example project now correctly reports spike indexes for the input group.

Please refer to the `full documentation <http://genn-team.github.io/genn/documentation/html/index.html>`__ for further details, tutorials and complete code documentation.

Release Notes for GeNN v2.2
===========================

This release includes minor new features, some core code improvements and several bug fixes on GeNN v2.1.

User Side Changes
~~~~~~~~~~~~~~~~~

#. GeNN now analyses automatically which parameters each kernel needs access to and these and only these are passed in the kernel argument list in addition to the global time t. These parameters can be a combination of extraGlobalNeuronKernelParameters and extraGlobalSynapseKernelParameters in either neuron or synapse kernel. In the unlikely case that users wish to call kernels directly, the correct call can be found in the ``stepTimeGPU()`` function.
   
   Reflecting these changes, the predefined Poisson neurons now simply have two extraGlobalNeuronParameter ``rates`` and ``offset`` which replace the previous custom pointer to the array of input rates and integer offset to indicate the current input pattern. These extraGlobalNeuronKernelParameters are passed to the neuron kernel automatically, but the rates themselves within the array are of course not updated automatically (this is exactly as before with the specifically generated kernel arguments for Poisson neurons).
   
   The concept of "directInput" has been removed. Users can easily achieve the same functionality by adding an additional variable (if there are individual inputs to neurons), an extraGlobalNeuronParameter (if the input is homogeneous but time dependent) or, obviously, a simple parameter if it's homogeneous and constant. The global time variable "t" is now provided by GeNN; please make sure that you are not duplicating its definition or shadowing it. This could have severe consequences for simulation correctness (e.g. time not advancing in cases of over-shadowing).

#. We introduced the namespace GENN_PREFERENCES which contains variables that determine the behaviour of GeNN.

#. We introduced a new code snippet called "supportCode" for neuron models, weightupdate models and post-synaptic models. This code snippet is intended to contain user-defined functions that are used from the other code snippets. We advise where possible to define the support code functions with the CUDA keywords "\__host\__ \__device\__" so that they are available for both GPU and CPU version. Alternatively one can define separate versions for **host** and **device** in the snippet. The snippets are automatically made available to the relevant code parts. This is regulated through namespaces so that name clashes between different models do not matter. An exception are hash defines. They can in principle be used in the supportCode snippet but need to be protected specifically using ifndef. For example
   
   .. ref-code-block:: cpp
   
   	#ifndef clip(x)
   	#define clip(x) x > 10.0? 10.0 : x
   	#endif
   
   If there are conflicting definitions for hash defines, the one that appears first in the GeNN generated code will then prevail.

#. The new convenience macros spikeCount_XX and spike_XX where "XX" is the name of the neuron group are now also available for events: spikeEventCount_XX and spikeEvent_XX. They access the values for the current time step even if there are synaptic delays and spikes events are stored in circular queues.

#. The old buildmodel.[sh\|bat] scripts have been superseded by new genn-buildmodel.[sh\|bat] scripts. These scripts accept UNIX style option switches, allow both relative and absolute model file paths, and allow the user to specify the directory in which all output files are placed (-o <path>). Debug (-d), CPU-only (-c) and show help (-h) are also defined.

#. We have introduced a CPU-only "-c" genn-buildmodel switch, which, if it's defined, will generate a GeNN version that is completely independent from CUDA and hence can be used on computers without CUDA installation or CUDA enabled hardware. Obviously, this then can also only run on CPU. CPU only mode can either be switched on by defining CPU_ONLY in the model description file or by passing appropriate parameters during the build, in particular
   
   .. ref-code-block:: cpp
   
   	genn-buildmodel.[sh|bat] \<modelfile\> -c
   	make release CPU_ONLY=1

#. The new genn-buildmodel "-o" switch allows the user to specify the output directory for all generated files - the default is the current directory. For example, a user project could be in '/home/genn_project', whilst the GeNN directory could be '/usr/local/genn'. The GeNN directory is kept clean, unless the user decides to build the sample projects inside of it without copying them elsewhere. This allows the deployment of GeNN to a read-only directory, like '/usr/local' or 'C:\Program Files'. It also allows multiple users - i.e. on a compute cluster - to use GeNN simultaneously, without overwriting each other's code-generation files, etcetera.

#. The ARM architecture is now supported - e.g. the NVIDIA Jetson development platform.

#. The NVIDIA CUDA SM_5\* (Maxwell) architecture is now supported.

#. An error is now thrown when the user tries to use double precision floating-point numbers on devices with architecture older than SM_13, since these devices do not support double precision.

#. All GeNN helper functions and classes, such as ``toString()`` and ``NNmodel``, are defined in the header files at ``genn/lib/include/``, for example ``stringUtils.h`` and ``modelSpec.h``, which should be individually included before the functions and classes may be used. The functions and classes are actually implementated in the static library ``genn\lib\lib\genn.lib`` (Windows) or ``genn/lib/lib/libgenn.a`` (Mac, Linux), which must be linked into the final executable if any GeNN functions or classes are used.

#. In the ``modelDefinition()`` file, only the header file ``modelSpec.h`` should be included - i.e. not the source file ``modelSpec.cc``. This is because the declaration and definition of ``NNmodel``, and associated functions, has been separated into ``modelSpec.h`` and ``modelSpec.cc``, respectively. This is to enable NNmodel code to be precompiled separately. Henceforth, only the header file ``modelSpec.h`` should be included in model definition files!

#. In the ``modelDefinition()`` file, DT is now preferrably defined using ``model.setDT(<val>);``, rather than # ``define DT <val>``, in order to prevent problems with DT macro redefinition. For backward-compatibility reasons, the old # ``define DT <val>`` method may still be used, however users are advised to adopt the new method.

#. In preparation for multi-GPU support in GeNN, we have separated out the compilation of generated code from user-side code. This will eventually allow us to optimise and compile different parts of the model with different CUDA flags, depending on the CUDA device chosen to execute that particular part of the model. As such, we have had to use a header file ``definitions.h`` as the generated code interface, rather than the ``runner.cc`` file. In practice, this means that user-side code should include ``myModel_CODE/definitions.h``, rather than ``myModel_CODE/runner.cc``. Including ``runner.cc`` will likely result in pages of linking errors at best!

Developer Side Changes
~~~~~~~~~~~~~~~~~~~~~~

#. Blocksize optimization and device choice now obtain the ptxas information on memory usage from a CUDA driver API call rather than from parsing ptxas output of the nvcc compiler. This adds robustness to any change in the syntax of the compiler output.

#. The information about device choice is now stored in variables in the namespace ``GENN_PREFERENCES``. This includes ``chooseDevice``, ``optimiseBlockSize``, ``optimizeCode``, ``debugCode``, ``showPtxInfo``, ``defaultDevice``. ``asGoodAsZero`` has also been moved into this namespace.

#. We have also introduced the namespace GENN_FLAGS that contains unsigned int variables that attach names to numeric flags that can be used within GeNN.

#. The definitions of all generated variables and functions such as pullXXXStateFromDevice etc, are now generated into definitions.h. This is useful where one wants to compile separate object files that cannot all include the full definitions in e.g. "runnerGPU.cc". One example where this is useful is the brian2genn interface.

#. A number of feature tests have been added that can be found in the ``featureTests`` directory. They can be run with the respective ``runTests.sh`` scripts. The ``cleanTests.sh`` scripts can be used to remove all generated code after testing.

Improvements
~~~~~~~~~~~~

#. Improved method of obtaining ptxas compiler information on register and shared memory usage and an improved algorithm for estimating shared memory usage requirements for different block sizes.

#. Replaced pageable CPU-side memory with `page-locked memory <https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/>`__. This can significantly speed up simulations in which a lot of data is regularly copied to and from a CUDA device.

#. GeNN library objects and the main generateALL binary objects are now compiled separately, and only when a change has been made to an object's source, rather than recompiling all software for a minor change in a single source file. This should speed up compilation in some instances.

Bug fixes:
~~~~~~~~~~

#. Fixed a minor bug with delayed synapses, where delaySlot is declared but not referenced.

#. We fixed a bug where on rare occasions a synchronisation problem occurred in sparse synapse populations.

#. We fixed a bug where the combined spike event condition from several synapse populations was not assembled correctly in the code generation phase (the parameter values of the first synapse population over-rode the values of all other populations in the combined condition).

Please refer to the `full documentation <http://genn-team.github.io/genn/documentation/html/index.html>`__ for further details, tutorials and complete code documentation.

Release Notes for GeNN v2.1
===========================

This release includes some new features and several bug fixes on GeNN v2.0.

User Side Changes
~~~~~~~~~~~~~~~~~

#. Block size debugging flag and the asGoodAsZero variables are moved into include/global.h.

#. NGRADSYNAPSES dynamics have changed (See Bug fix #4) and this change is applied to the example projects. If you are using this synapse model, you may want to consider changing model parameters.

#. The delay slots are now such that NO_DELAY is 0 delay slots (previously 1) and 1 means an actual delay of 1 time step.

#. The convenience function convertProbabilityToRandomNumberThreshold(float \*, uint64_t \*, int) was changed so that it actually converts firing probability/timestep into a threshold value for the GeNN random number generator (as its name always suggested). The previous functionality of converting a *rate* in kHz into a firing threshold number for the GeNN random number generator is now provided with the name convertRateToRandomNumberThreshold(float \*, uint64_t \*, int)

#. Every model definition function ``modelDefinition()`` now needs to end with calling ``:ref:`NNmodel::finalize() <doxid-da/dfd/classModelSpec_1ad5166bfbc1a19f2d829be2ed1d8973cc>``` for the defined network model. This will lock down the model and prevent any further changes to it by the supported methods. It also triggers necessary analysis of the model structure that should only be performed once. If the ``finalize()`` function is not called, GeNN will issue an error and exit before code generation.

#. To be more consistent in function naming the ``pull\<SYNAPSENAME\>FromDevice`` and ``push\<SYNAPSENAME\>ToDevice`` have been renamed to ``pull\<SYNAPSENAME\>StateFromDevice`` and ``push\<SYNAPSENAME\>StateToDevice``. The old versions are still supported through macro definitions to make the transition easier.

#. New convenience macros are now provided to access the current spike numbers and identities of neurons that spiked. These are called spikeCount_XX and spike_XX where "XX" is the name of the neuron group. They access the values for the current time step even if there are synaptic delays and spikes are stored in circular queues.

#. There is now a pre-defined neuron type "SPIKECOURCE" which is empty and can be used to define PyNN style spike source arrays.

#. The macros FLOAT and DOUBLE were replaced with GENN_FLOAT and GENN_DOUBLE due to name clashes with typedefs in Windows that define FLOAT and DOUBLE.

Developer Side Changes
~~~~~~~~~~~~~~~~~~~~~~

#. We introduced a file definitions.h, which is generated and filled with useful macros such as spkQuePtrShift which tells users where in the circular spike queue their spikes start.

Improvements
~~~~~~~~~~~~

#. Improved debugging information for block size optimisation and device choice.

#. Changed the device selection logic so that device occupancy has larger priority than device capability version.

#. A new HH model called TRAUBMILES_PSTEP where one can set the number of inner loops as a parameter is introduced. It uses the TRAUBMILES_SAFE method.

#. An alternative method is added for the insect olfaction model in order to fix the number of connections to a maximum of 10K in order to avoid negative conductance tails.

#. We introduced a preprocessor define directive for an "int\_" function that translates floating points to integers.

Bug fixes:
~~~~~~~~~~

#. AtomicAdd replacement for old GPUs were used by mistake if the model runs in double precision.

#. Timing of individual kernels is fixed and improved.

#. More careful setting of maximum number of connections in sparse connectivity, covering mixed dense/sparse network scenarios.

#. NGRADSYNAPSES was not scaling correctly with varying time step.

#. Fixed a bug where learning kernel with sparse connectivity was going out of range in an array.

#. Fixed synapse kernel name substitutions where the "dd\_" prefix was omitted by mistake.

Please refer to the `full documentation <http://genn-team.github.io/genn/documentation/html/index.html>`__ for further details, tutorials and complete code documentation.

Release Notes for GeNN v2.0
===========================

Version 2.0 of GeNN comes with a lot of improvements and added features, some of which have necessitated some changes to the structure of parameter arrays among others.

User Side Changes
~~~~~~~~~~~~~~~~~

#. Users are now required to call ``initGeNN()`` in the model definition function before adding any populations to the neuronal network model.

#. glbscnt is now call glbSpkCnt for consistency with glbSpkEvntCnt.

#. There is no longer a privileged parameter ``Epre``. Spike type events are now defined by a code string ``spkEvntThreshold``, the same way proper spikes are. The only difference is that Spike type events are specific to a synapse type rather than a neuron type.

#. The function setSynapseG has been deprecated. In a ``GLOBALG`` scenario, the variables of a synapse group are set to the initial values provided in the ``modeldefinition`` function.

#. Due to the split of synaptic models into weightUpdateModel and postSynModel, the parameter arrays used during model definition need to be carefully split as well so that each side gets the right parameters. For example, previously
   
   .. ref-code-block:: cpp
   
   	float myPNKC_p[3]= {
   	0.0,           // 0 - Erev: Reversal potential
   	-20.0,         // 1 - Epre: Presynaptic threshold potential
   	1.0            // 2 - tau_S: decay time constant for S [ms]
   	};
   
   would define the parameter array of three parameters, ``Erev``, ``Epre``, and ``tau_S`` for a synapse of type ``NSYNAPSE``. This now needs to be "split" into
   
   .. ref-code-block:: cpp
   
   	float *myPNKC_p= NULL;
   	float postExpPNKC[2]={
   	  1.0,            // 0 - tau_S: decay time constant for S [ms]
   	  0.0         // 1 - Erev: Reversal potential
   	};
   
   i.e. parameters ``Erev`` and ``tau_S`` are moved to the post-synaptic model and its parameter array of two parameters. ``Epre`` is discontinued as a parameter for ``NSYNAPSE``. As a consequence the weightupdate model of ``NSYNAPSE`` has no parameters and one can pass ``NULL`` for the parameter array in ``addSynapsePopulation``. The correct parameter lists for all defined neuron and synapse model types are listed in the `User Manual <http://genn-team.github.io/genn/documentation/html/dc/d05/UserManual.html>`__. If the parameters are not redefined appropriately this will lead to uncontrolled behaviour of models and likely to segmentation faults and crashes.

#. Advanced users can now define variables as type ``scalar`` when introducing new neuron or synapse types. This will at the code generation stage be translated to the model's floating point type (ftype), ``float`` or ``double``. This works for defining variables as well as in all code snippets. Users can also use the expressions ``SCALAR_MAX`` and ``SCALAR_MIN`` for ``FLT_MIN``, ``FLT_MAX``, ``DBL_MIN`` and ``DBL_MAX``, respectively. Corresponding definitions of ``scalar``, ``SCALAR_MIN`` and ``SCALAR_MAX`` are also available for user-side code whenever the code-generated file ``runner.cc`` has been included.

#. The example projects have been re-organized so that wrapper scripts of the ``generate_run`` type are now all located together with the models they run instead of in a common ``tools`` directory. Generally the structure now is that each example project contains the wrapper script ``generate_run`` and a ``model`` subdirectory which contains the model description file and the user side code complete with Makefiles for Unix and Windows operating systems. The generated code will be deposited in the ``model`` subdirectory in its own ``modelname_CODE`` folder. Simulation results will always be deposited in a new sub-folder of the main project directory.

#. The ``addSynapsePopulation(...)`` function has now more mandatory parameters relating to the introduction of separate weightupdate models (pre-synaptic models) and postynaptic models. The correct syntax for the ``addSynapsePopulation(...)`` can be found with detailed explanations in teh `User Manual <http://genn-team.github.io/genn/documentation/html/dc/d05/UserManual.html>`__.

#. We have introduced a simple performance profiling method that users can employ to get an overview over the differential use of time by different kernels. To enable the timers in GeNN generated code, one needs to declare
   
   .. ref-code-block:: cpp
   
   	networkmodel.setTiming(TRUE);
   
   This will make available and operate GPU-side cudeEvent based timers whose cumulative value can be found in the double precision variables ``neuron_tme``, ``synapse_tme`` and ``learning_tme``. They measure the accumulated time that has been spent calculating the neuron kernel, synapse kernel and learning kernel, respectively. CPU-side timers for the simulation functions are also available and their cumulative values can be obtained through
   
   .. ref-code-block:: cpp
   
   	float x= sdkGetTimerValue(&neuron_timer);
   	float y= sdkGetTimerValue(&synapse_timer);
   	float z= sdkGetTimerValue(&learning_timer);
   
   The :ref:`Insect olfaction model <doxid-d9/d61/Examples_1ex_mbody>` example shows how these can be used in the user-side code. To enable timing profiling in this example, simply enable it for GeNN:
   
   .. ref-code-block:: cpp
   
   	model.setTiming(TRUE);
   
   in ``MBody1.cc`` 's ``modelDefinition`` function and define the macro ``TIMING`` in ``classol_sim.h``
   
   .. ref-code-block:: cpp
   
   	#define TIMING
   
   This will have the effect that timing information is output into ``OUTNAME_output/OUTNAME.timingprofile``.

Developer Side Changes
~~~~~~~~~~~~~~~~~~~~~~

#. ``allocateSparseArrays()`` has been changed to take the number of connections, connN, as an argument rather than expecting it to have been set in the Connetion struct before the function is called as was the arrangement previously.

#. For the case of sparse connectivity, there is now a reverse mapping implemented with revers index arrays and a remap array that points to the original positions of variable values in teh forward array. By this mechanism, revers lookups from post to pre synaptic indices are possible but value changes in the sparse array values do only need to be done once.

#. SpkEvnt code is no longer generated whenever it is not actually used. That is also true on a somewhat finer granularity where variable queues for synapse delays are only maintained if the corresponding variables are used in synaptic code. True spikes on the other hand are always detected in case the user is interested in them.

Please refer to the `full documentation <http://genn-team.github.io/genn/documentation/html/index.html>`__ for further details, tutorials and complete code documentation.

:ref:`Previous <doxid-d0/d81/PyGeNN>` \| :ref:`Top <doxid-df/ddb/ReleaseNotes>` \| :ref:`Next <doxid-dc/d05/UserManual>`

