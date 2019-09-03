.. index:: pair: page; Current source models
.. _doxid-d0/d1e/sectCurrentSourceModels:

Current source models
=====================

There is a number of predefined models which can be used with the :ref:`ModelSpec::addCurrentSource <doxid-da/dfd/classModelSpec_1aaf260ae8ffd52473b61a27974867c3e3>` function:

* :ref:`CurrentSourceModels::DC <doxid-d7/da1/classCurrentSourceModels_1_1DC>`

* :ref:`CurrentSourceModels::GaussianNoise <doxid-d0/d7d/classCurrentSourceModels_1_1GaussianNoise>`



.. _doxid-d0/d1e/sectCurrentSourceModels_1sect_own_current_source:

Defining your own current source model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to define a new current source type for use in a GeNN application, it is necessary to define a new class derived from :ref:`CurrentSourceModels::Base <doxid-d0/de0/classCurrentSourceModels_1_1Base>`. For convenience the methods this class should implement can be implemented using macros:

* :ref:`DECLARE_MODEL(TYPE, NUM_PARAMS, NUM_VARS) <doxid-d4/d13/models_8h_1ae0c817e85c196f39cf62d608883cda13>`, :ref:`SET_DERIVED_PARAMS() <doxid-de/d6c/snippet_8h_1aa592bfe3ce05ffc19a8f21d8482add6b>`, :ref:`SET_PARAM_NAMES() <doxid-de/d6c/snippet_8h_1a75315265035fd71c5b5f7d7f449edbd7>`, :ref:`SET_VARS() <doxid-d4/d13/models_8h_1a3025b9fc844fccdf8cc2b51ef4a6e0aa>` perform the same roles as they do in the neuron models discussed in :ref:`Defining your own neuron type <doxid-de/ded/sectNeuronModels_1sect_own>`.

* :ref:`SET_INJECTION_CODE(INJECTION_CODE) <doxid-da/d49/currentSourceModels_8h_1adf53ca7b56294cfcca6f22ddfd1daf4f>` : where INJECTION_CODE contains the code for injecting current into the neuron every simulation timestep. The $(injectCurrent, ) function is used to inject current.

For example, using these macros, we can define a uniformly distributed noisy current source:

.. ref-code-block:: cpp

	class UniformNoise : public :ref:`CurrentSourceModels::Base <doxid-d0/de0/classCurrentSourceModels_1_1Base>`
	{
	public:
	    :ref:`DECLARE_MODEL <doxid-d4/d13/models_8h_1ae0c817e85c196f39cf62d608883cda13>`(UniformNoise, 1, 0);
	    
	    :ref:`SET_SIM_CODE <doxid-d3/dc0/neuronModels_8h_1a8d014c818d8ee68f3a16838dcd4f030f>`("$(injectCurrent, $(gennrand_uniform) * $(magnitude));");
	    
	    :ref:`SET_PARAM_NAMES <doxid-de/d6c/snippet_8h_1a75315265035fd71c5b5f7d7f449edbd7>`({"magnitude"});
	};

:ref:`Previous <doxid-dd/de4/sect_postsyn>` \| :ref:`Top <doxid-dc/d05/UserManual>` \| :ref:`Next <doxid-d5/d39/subsect34>`

