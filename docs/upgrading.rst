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

- single PSM code string
- add_synapse_population
- The row build and diagonal build state variables in sparse/toeplitz connectivity building code were really ugly and confusing. Sparse connectivity init snippets now just let the user write whatever sort of loop they want and do the initialisation outside and toeplitz reuses the for_each_synapse structure described above to do similar.
- ``GLOBALG`` and ``INDIVIDUALG`` confused almost all new users and were really only used with ``StaticPulse`` weight update models. Same functionality can be achieved with a ``StaticPulseConstantWeight`` version with the weight as a parameter. Then I've renamed all the 'obvious' :class:`.SynapseMatrixType` variants so you just chose :attr:`.SynapseMatrixType.SPARSE`, :attr:`.SynapseMatrixType.DENSE`, :attr:`.SynapseMatrixType.TOEPLITZ` or :attr:`.SynapseMatrixType.PROCEDURAL` (with :attr:`.SynapseMatrixType.DENSE_PROCEDURALG` and :attr:`.SynapseMatrixType.PROCEDURAL_KERNELG` for more unusual options)
- Extra global parameters only support the 'pointer' form, awaiting a PR to implement settable parameters to replace the other sort
- Removed implicit variable referneces