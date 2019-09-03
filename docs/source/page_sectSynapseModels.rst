.. index:: pair: page; Weight update models
.. _doxid-d5/d24/sectSynapseModels:

Weight update models
====================

Currently 4 predefined weight update models are available:

* :ref:`WeightUpdateModels::StaticPulse <doxid-d9/d74/classWeightUpdateModels_1_1StaticPulse>`

* :ref:`WeightUpdateModels::StaticPulseDendriticDelay <doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay>`

* :ref:`WeightUpdateModels::StaticGraded <doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded>`

* :ref:`WeightUpdateModels::PiecewiseSTDP <doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP>`

For more details about these built-in synapse models, see :ref:`Nowotny2010 <doxid-d0/de3/citelist_1CITEREF_Nowotny2010>`.



.. _doxid-d5/d24/sectSynapseModels_1sect34:

Defining a new weight update model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like the neuron models discussed in :ref:`Defining your own neuron type <doxid-de/ded/sectNeuronModels_1sect_own>`, new weight update models are created by defining a class. Weight update models should all be derived from WeightUpdateModel::Base and, for convenience, the methods a new weight update model should implement can be implemented using macros:

* :ref:`SET_DERIVED_PARAMS() <doxid-de/d6c/snippet_8h_1aa592bfe3ce05ffc19a8f21d8482add6b>`, :ref:`SET_PARAM_NAMES() <doxid-de/d6c/snippet_8h_1a75315265035fd71c5b5f7d7f449edbd7>`, :ref:`SET_VARS() <doxid-d4/d13/models_8h_1a3025b9fc844fccdf8cc2b51ef4a6e0aa>` and :ref:`SET_EXTRA_GLOBAL_PARAMS() <doxid-de/d51/initSparseConnectivitySnippet_8h_1aa33e3634a531794ddac1ad49bde09071>` perform the same roles as they do in the neuron models discussed in :ref:`Defining your own neuron type <doxid-de/ded/sectNeuronModels_1sect_own>`.

* :ref:`DECLARE_WEIGHT_UPDATE_MODEL(TYPE, NUM_PARAMS, NUM_VARS, NUM_PRE_VARS, NUM_POST_VARS) <doxid-dc/dab/weightUpdateModels_8h_1a611a9113f742a9d07d3def4298a0ea68>` is an extended version of ``:ref:`DECLARE_MODEL() <doxid-d4/d13/models_8h_1ae0c817e85c196f39cf62d608883cda13>``` which declares the boilerplate code required for a weight update model with pre and postsynaptic as well as per-synapse state variables.

* :ref:`SET_PRE_VARS() <doxid-dc/dab/weightUpdateModels_8h_1a9a2bc6f56fa2bfb7008e915710720cfd>` and :ref:`SET_POST_VARS() <doxid-dc/dab/weightUpdateModels_8h_1a906e656a5980ea57c9f1b7c3185e876b>` define state variables associated with pre or postsynaptic neurons rather than synapses. These are typically used to efficiently implement *trace* variables for use in STDP learning rules :ref:`Morrison2008 <doxid-d0/de3/citelist_1CITEREF_Morrison2008>`. Like other state variables, variables defined here as ``NAME`` can be accessed in weight update model code strings using the $(NAME) syntax.

* :ref:`SET_SIM_CODE(SIM_CODE) <doxid-dc/dab/weightUpdateModels_8h_1a8d014c818d8ee68f3a16838dcd4f030f>` : defines the simulation code that is used when a true spike is detected. The update is performed only in timesteps after a neuron in the presynaptic population has fulfilled its threshold detection condition. Typically, spikes lead to update of synaptic variables that then lead to the activation of input into the post-synaptic neuron. Most of the time these inputs add linearly at the post-synaptic neuron. This is assumed in GeNN and the term to be added to the activation of the post-synaptic neuron should be applied using the the $(addToInSyn, weight) function. For example
  
  .. ref-code-block:: cpp
  
  	:ref:`SET_SIM_CODE <doxid-d3/dc0/neuronModels_8h_1a8d014c818d8ee68f3a16838dcd4f030f>`(
  	    "$(addToInSyn, $(inc));\n"
  
  where "inc" is the increment of the synaptic input to a post-synaptic neuron for each pre-synaptic spike. The simulation code also typically contains updates to the internal synapse variables that may have contributed to . For an example, see :ref:`WeightUpdateModels::StaticPulse <doxid-d9/d74/classWeightUpdateModels_1_1StaticPulse>` for a simple synapse update model and :ref:`WeightUpdateModels::PiecewiseSTDP <doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP>` for a more complicated model that uses STDP. To apply input to the post-synaptic neuron with a dendritic (i.e. between the synapse and the postsynaptic neuron) delay you can instead use the $(addToInSynDelay, weight, delay) function. For example
  
  .. ref-code-block:: cpp
  
  	:ref:`SET_SIM_CODE <doxid-d3/dc0/neuronModels_8h_1a8d014c818d8ee68f3a16838dcd4f030f>`(
  	    "$(addToInSynDelay, $(inc), $(delay));");
  
  where, once again, ``inc`` is the magnitude of the input step to apply and ``delay`` is the length of the dendritic delay in timesteps. By implementing ``delay`` as a weight update model variable, heterogeneous synaptic delays can be implemented. For an example, see :ref:`WeightUpdateModels::StaticPulseDendriticDelay <doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay>` for a simple synapse update model with heterogeneous dendritic delays. When using dendritic delays, the **maximum** dendritic delay for a synapse populations must be specified using the ``:ref:`SynapseGroup::setMaxDendriticDelayTimesteps() <doxid-dc/dfa/classSynapseGroup_1a220307d4043e8bf1bed07552829f2a17>``` function.

* :ref:`SET_EVENT_THRESHOLD_CONDITION_CODE(EVENT_THRESHOLD_CONDITION_CODE) <doxid-dc/dab/weightUpdateModels_8h_1a9e0fecc624daee536a388777788cd9de>` defines a condition for a synaptic event. This typically involves the pre-synaptic variables, e.g. the membrane potential:
  
  .. ref-code-block:: cpp
  
  	:ref:`SET_EVENT_THRESHOLD_CONDITION_CODE <doxid-dc/dab/weightUpdateModels_8h_1a9e0fecc624daee536a388777788cd9de>`("$(V_pre) > -0.02");
  
  Whenever this expression evaluates to true, the event code set using the :ref:`SET_EVENT_CODE() <doxid-dc/dab/weightUpdateModels_8h_1a8159c6f595e865d6bf609f045c07361e>` macro is executed. For an example, see :ref:`WeightUpdateModels::StaticGraded <doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded>`.

* :ref:`SET_EVENT_CODE(EVENT_CODE) <doxid-dc/dab/weightUpdateModels_8h_1a8159c6f595e865d6bf609f045c07361e>` defines the code that is used when the event threshold condition is met (as set using the :ref:`SET_EVENT_THRESHOLD_CONDITION_CODE() <doxid-dc/dab/weightUpdateModels_8h_1a9e0fecc624daee536a388777788cd9de>` macro).

* :ref:`SET_LEARN_POST_CODE(LEARN_POST_CODE) <doxid-dc/dab/weightUpdateModels_8h_1a9f1ad825b90bcbaab3b0d18fc4d00016>` defines the code which is used in the learnSynapsesPost kernel/function, which performs updates to synapses that are triggered by post-synaptic spikes. This is typically used in STDP-like models e.g. :ref:`WeightUpdateModels::PiecewiseSTDP <doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP>`.

* :ref:`SET_SYNAPSE_DYNAMICS_CODE(SYNAPSE_DYNAMICS_CODE) <doxid-dc/dab/weightUpdateModels_8h_1ae68b0e54b73f5cda5fe9bab3667de3a8>` defines code that is run for each synapse, each timestep i.e. unlike the others it is not event driven. This can be used where synapses have internal variables and dynamics that are described in continuous time, e.g. by ODEs. However using this mechanism is typically computationally very costly because of the large number of synapses in a typical network. By using the $(addtoinsyn), $(updatelinsyn) and $(addToDenDelay) mechanisms discussed in the context of :ref:`SET_SIM_CODE() <doxid-d3/dc0/neuronModels_8h_1a8d014c818d8ee68f3a16838dcd4f030f>`, the synapse dynamics can also be used to implement continuous synapses for rate-based models.

* :ref:`SET_PRE_SPIKE_CODE() <doxid-dc/dab/weightUpdateModels_8h_1aede2f97f853841236f081c8d7b9d156f>` and :ref:`SET_POST_SPIKE_CODE() <doxid-dc/dab/weightUpdateModels_8h_1aef99e5858038673e6b268f4af07864c2>` define code that is called whenever there is a pre or postsynaptic spike. Typically these code strings are used to update any pre or postsynaptic state variables.

* :ref:`SET_NEEDS_PRE_SPIKE_TIME(PRE_SPIKE_TIME_REQUIRED) <doxid-dc/dab/weightUpdateModels_8h_1ad06378df00a5d9ffe4068ba2c01b09ab>` and :ref:`SET_NEEDS_POST_SPIKE_TIME(POST_SPIKE_TIME_REQUIRED) <doxid-dc/dab/weightUpdateModels_8h_1a4f3e008922887cba8cfafc0fb0e53965>` define whether the weight update needs to know the times of the spikes emitted from the pre and postsynaptic populations. For example an STDP rule would be likely to require:
  
  .. ref-code-block:: cpp
  
  	:ref:`SET_NEEDS_PRE_SPIKE_TIME <doxid-dc/dab/weightUpdateModels_8h_1ad06378df00a5d9ffe4068ba2c01b09ab>`(true);
  	:ref:`SET_NEEDS_POST_SPIKE_TIME <doxid-dc/dab/weightUpdateModels_8h_1a4f3e008922887cba8cfafc0fb0e53965>`(true);

All code snippets, aside from those defined with ``:ref:`SET_PRE_SPIKE_CODE() <doxid-dc/dab/weightUpdateModels_8h_1aede2f97f853841236f081c8d7b9d156f>``` and ``:ref:`SET_POST_SPIKE_CODE() <doxid-dc/dab/weightUpdateModels_8h_1aef99e5858038673e6b268f4af07864c2>```, can be used to manipulate any synapse variable and so learning rules can combine both time-drive and event-driven processes.

:ref:`Previous <doxid-de/ded/sectNeuronModels>` \| :ref:`Top <doxid-dc/d05/UserManual>` \| :ref:`Next <doxid-dd/de4/sect_postsyn>`

