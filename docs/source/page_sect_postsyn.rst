.. index:: pair: page; Postsynaptic integration methods
.. _doxid-dd/de4/sect_postsyn:

Postsynaptic integration methods
================================

There are currently 3 built-in postsynaptic integration methods:

* :ref:`PostsynapticModels::ExpCurr <doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr>`

* :ref:`PostsynapticModels::ExpCond <doxid-d5/d27/classPostsynapticModels_1_1ExpCond>`

* :ref:`PostsynapticModels::DeltaCurr <doxid-de/da9/classPostsynapticModels_1_1DeltaCurr>`



.. _doxid-dd/de4/sect_postsyn_1sect_new_postsynaptic:

Defining a new postsynaptic model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The postsynaptic model defines how synaptic activation translates into an input current (or other input term for models that are not current based). It also can contain equations defining dynamics that are applied to the (summed) synaptic activation, e.g. an exponential decay over time.

In the same manner as to both the neuron and weight update models discussed in :ref:`Defining your own neuron type <doxid-de/ded/sectNeuronModels_1sect_own>` and :ref:`Defining a new weight update model <doxid-d5/d24/sectSynapseModels_1sect34>`, postsynamic model definitions are encapsulated in a class derived from :ref:`PostsynapticModels::Base <doxid-d1/d3a/classPostsynapticModels_1_1Base>`. Again, the methods that a postsynaptic model should implement can be implemented using the following macros:

* :ref:`DECLARE_MODEL(TYPE, NUM_PARAMS, NUM_VARS) <doxid-d4/d13/models_8h_1ae0c817e85c196f39cf62d608883cda13>`, :ref:`SET_DERIVED_PARAMS() <doxid-de/d6c/snippet_8h_1aa592bfe3ce05ffc19a8f21d8482add6b>`, :ref:`SET_PARAM_NAMES() <doxid-de/d6c/snippet_8h_1a75315265035fd71c5b5f7d7f449edbd7>`, :ref:`SET_VARS() <doxid-d4/d13/models_8h_1a3025b9fc844fccdf8cc2b51ef4a6e0aa>` perform the same roles as they do in the neuron models discussed in :ref:`Defining your own neuron type <doxid-de/ded/sectNeuronModels_1sect_own>`.

* :ref:`SET_DECAY_CODE(DECAY_CODE) <doxid-d8/d47/postsynapticModels_8h_1a76caa98a308e162345cee61e025de022>` defines the code which provides the continuous time dynamics for the summed presynaptic inputs to the postsynaptic neuron. This usually consists of some kind of decay function.

* :ref:`SET_APPLY_INPUT_CODE(APPLY_INPUT_CODE) <doxid-d8/d47/postsynapticModels_8h_1a41d7141aeae91e2840c2629106b6a3b1>` defines the code specifying the conversion from synaptic inputs to a postsynaptic neuron input current. e.g. for a conductance model:
  
  .. ref-code-block:: cpp
  
  	:ref:`SET_APPLY_INPUT_CODE <doxid-d8/d47/postsynapticModels_8h_1a41d7141aeae91e2840c2629106b6a3b1>`("$(Isyn) += $(inSyn) * ($(E) - $(V))");
  
  where $(E) is a postsynaptic model parameter specifying reversal potential and $(V) is the variable containing the postsynaptic neuron's membrane potential. As discussed in :ref:`Built-in Variables in GeNN <doxid-d0/da6/UserGuide_1predefinedVars>`, $(Isyn) is the built in variable used to sum neuron input. However additional input variables can be added to a neuron model using the :ref:`SET_ADDITIONAL_INPUT_VARS() <doxid-d3/dc0/neuronModels_8h_1a96a3e23f5c7309a47bc6562e0be81e99>` macro (see :ref:`Defining your own neuron type <doxid-de/ded/sectNeuronModels_1sect_own>` for more details).

:ref:`Previous <doxid-de/ded/sectNeuronModels>` \| :ref:`Top <doxid-dc/d05/UserManual>` \| :ref:`Next <doxid-d0/d1e/sectCurrentSourceModels>`

