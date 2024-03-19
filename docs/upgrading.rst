.. py:currentmodule:: pygenn

=====================
Upgrading from GeNN 4
=====================
GeNN 5 does not aim to be a backward-compatible upgrade.
In PyGeNN we have strived to add backward compatibility where possible but most models will need to be updated.

--------
GeNNCode
--------
In GeNN 4 GeNNCode was passed directly to the underlying compiler whereas, in GeNN 5, it is parsed and only a subset of C99 language features are supported.
In PyGeNN,

- The & operator isn't supported - on the target GPU hardware, local variables are assumed to be stored in registers and not addressable. The only time this is limiting is when dealing with extra global parameter arrays as you can no longer do stuff like ``const int *egpSubset = &egp[offset];`` and instead have to do ``const int *egpSubset = egp + offset;``.
- The old ``$(xx)`` syntax for referencing GeNN stuff is no longer necessary at all. 

--------------
Syntax changes
--------------
In order to streamline the modelling process and reduce the number of ways to achieve the same thing we have to maintain,
several areas of GeNN syntax have changed:

- Postsynaptic models were previously implemented with two code strings: ``apply_input_code`` and
  ``decay_code``. This was unnecessarily complex and the order these were evaluated in wasn't obvious.
  Therefore, these have been replaced by a single ``sim_code`` code string from which input can be
  delivered to the postsynaptic neuron using the ``injectCurrent(x)`` function.
  
- :meth:`.GeNNModel.add_synapse_population` was becoming rather cumbersome and this was only made worse by postsynaptic models and weight update models now taking variable references.
  To improve this:

    * Axonal delay is now no longer a required parameter and, if required, can now be set using :attr:`.SynapseGroup.axonal_delay_steps`
    * The postsynaptic and weight update models used by the synapse group are initialised seperately using :func:`.init_postsynaptic` and :func:`.init_weight_update` respectively.

- The row build and diagonal build state variables in sparse/toeplitz connectivity building code were really ugly and confusing. Sparse connectivity init snippets now just let the user write whatever sort of loop they want and do the initialisation outside and toeplitz reuses the for_each_synapse structure described above to do similar.
- ``GLOBALG`` and ``INDIVIDUALG`` confused almost all new users and were really only used with ``StaticPulse`` weight update models. Same functionality can be achieved with a ``StaticPulseConstantWeight`` version with the weight as a parameter. Then I've renamed all the 'obvious' :class:`.SynapseMatrixType` variants so you just chose :attr:`.SynapseMatrixType.SPARSE`, :attr:`.SynapseMatrixType.DENSE`, :attr:`.SynapseMatrixType.TOEPLITZ` or :attr:`.SynapseMatrixType.PROCEDURAL` (with :attr:`.SynapseMatrixType.DENSE_PROCEDURALG` and :attr:`.SynapseMatrixType.PROCEDURAL_KERNELG` for more unusual options)
- Extra global parameters used to be creatable with pointer types to allow models to access
  arbitrary sized arrays and non-pointer types to create parameters that could be modified at runtime.
  The latter use case is now handled much more efficiently using the `Dynamic parameters`_ system.
- Weight update and postsynaptic models could previous refer to neuron variables via _implicit_
  references. For example ``$(V_pre)`` would refer to a variable called ``V`` on the presynaptic
  neuron population. This mechanism has been replaced by adding _explicit_ references to
  weight update, postsynaptic and current source model (see `Variables references`_).
- It used to be possible to interact directly with the underlying spike data structures 
  used by GeNN to inject or record spikes. However, this was very innefficient and is no
  longer supported. Even if you need to access recorded spikes every timestep, using the 
  `Spike recording`_ system is still more efficient and to inject spikes we recommend using
  either the built in ``SpikeSourceArray`` model (as described in `Extra global parameters`_) 
  or custom `Neuron models`_.