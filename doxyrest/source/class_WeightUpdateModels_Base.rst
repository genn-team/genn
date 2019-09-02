.. index:: pair: class; WeightUpdateModels::Base
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base:

class WeightUpdateModels::Base
==============================

.. toctree::
	:hidden:

Overview
~~~~~~~~

:ref:`Base <doxid-d2/d05/classWeightUpdateModels_1_1Base>` class for all weight update models. :ref:`More...<details-d2/d05/classWeightUpdateModels_1_1Base>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <weightUpdateModels.h>
	
	class Base: public :ref:`Models::Base<doxid-d6/d97/classModels_1_1Base>`
	{
	public:
		// methods
	
		virtual std::string :ref:`getSimCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a0b7445981ce7bf71e7866fd961029004>`() const;
		virtual std::string :ref:`getEventCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a8c4939c38b32ae603cd237f0e8d76b8a>`() const;
		virtual std::string :ref:`getLearnPostCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a0bb39d77c70d759d9036352d316ee044>`() const;
		virtual std::string :ref:`getSynapseDynamicsCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1ab3daed63a1d17897aa73c741b728ea6e>`() const;
		virtual std::string :ref:`getEventThresholdConditionCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1aab9670aee177fafc6908f177b322b791>`() const;
		virtual std::string :ref:`getSimSupportCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a5ed9cae169e9808c6c8823e624880451>`() const;
		virtual std::string :ref:`getLearnPostSupportCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1ac5d1d2d7524cab0f19e965159dd58e8b>`() const;
		virtual std::string :ref:`getSynapseDynamicsSuppportCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1aca2d11a28a6cb587dba5f7ae9c87c445>`() const;
		virtual std::string :ref:`getPreSpikeCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a09e5ecd955d9a89bb8deeb5858fa718a>`() const;
		virtual std::string :ref:`getPostSpikeCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a2eab2ca9adfa8698ffe90392b41d1435>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getPreVars<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a59c2e29f7c607d87d9342ee88153013d>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getPostVars<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a9d81ca1fb2686a808e975f974ec4884d>`() const;
		virtual bool :ref:`isPreSpikeTimeRequired<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a26c3071dfdf87eaddb857a535894bf7a>`() const;
		virtual bool :ref:`isPostSpikeTimeRequired<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a02fb269c52929c962bab49d86d2ca45e>`() const;
		size_t :ref:`getPreVarIndex<doxid-d2/d05/classWeightUpdateModels_1_1Base_1add432f1a452d82183e0574d1fe171f75>`(const std::string& varName) const;
		size_t :ref:`getPostVarIndex<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a4bca317ba20ee97433d03930081deac3>`(const std::string& varName) const;
	};

	// direct descendants

	class :ref:`PiecewiseSTDP<doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP>`;
	class :ref:`StaticGraded<doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded>`;
	class :ref:`StaticPulse<doxid-d9/d74/classWeightUpdateModels_1_1StaticPulse>`;
	class :ref:`StaticPulseDendriticDelay<doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay>`;

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// typedefs
	
		typedef std::vector<std::string> :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>`;
		typedef std::vector<:ref:`EGP<doxid-dd/d5d/structSnippet_1_1Base_1_1EGP>`> :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>`;
		typedef std::vector<:ref:`ParamVal<doxid-d7/dda/structSnippet_1_1Base_1_1ParamVal>`> :ref:`ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>`;
		typedef std::vector<:ref:`DerivedParam<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam>`> :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>`;
		typedef std::vector<:ref:`Var<doxid-d5/d42/structModels_1_1Base_1_1Var>`> :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>`;

		// structs
	
		struct :ref:`DerivedParam<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam>`;
		struct :ref:`EGP<doxid-dd/d5d/structSnippet_1_1Base_1_1EGP>`;
		struct :ref:`ParamVal<doxid-d7/dda/structSnippet_1_1Base_1_1ParamVal>`;
		struct :ref:`Var<doxid-d5/d42/structModels_1_1Base_1_1Var>`;

		// methods
	
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1aad4f3bb00c5f29cb9d0e3585db3f4e20>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1a450c7783570d875e19bcd8a88d10bbf6>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d6/d97/classModels_1_1Base_1a5da12b4e51f0b969510dd97d45ad285a>`() const;
		virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` :ref:`getExtraGlobalParams<doxid-d6/d97/classModels_1_1Base_1ad6a043bb48b7620c4294854c042e561e>`() const;
		size_t :ref:`getVarIndex<doxid-d6/d97/classModels_1_1Base_1ab54e5508872ef8d1558b7da8aa25bb63>`(const std::string& varName) const;
		size_t :ref:`getExtraGlobalParamIndex<doxid-d6/d97/classModels_1_1Base_1a693ad5cfedde6e2db10200501c549c81>`(const std::string& paramName) const;

.. _details-d2/d05/classWeightUpdateModels_1_1Base:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`Base <doxid-d2/d05/classWeightUpdateModels_1_1Base>` class for all weight update models.

Methods
-------

.. index:: pair: function; getSimCode
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a0b7445981ce7bf71e7866fd961029004:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSimCode() const

Gets simulation code run when 'true' spikes are received.

.. index:: pair: function; getEventCode
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a8c4939c38b32ae603cd237f0e8d76b8a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getEventCode() const

Gets code run when events (all the instances where event threshold condition is met) are received.

.. index:: pair: function; getLearnPostCode
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a0bb39d77c70d759d9036352d316ee044:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getLearnPostCode() const

Gets code to include in the learnSynapsesPost kernel/function.

For examples when modelling STDP, this is where the effect of postsynaptic spikes which occur *after* presynaptic spikes are applied.

.. index:: pair: function; getSynapseDynamicsCode
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1ab3daed63a1d17897aa73c741b728ea6e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSynapseDynamicsCode() const

Gets code for synapse dynamics which are independent of spike detection.

.. index:: pair: function; getEventThresholdConditionCode
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1aab9670aee177fafc6908f177b322b791:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getEventThresholdConditionCode() const

Gets codes to test for events.

.. index:: pair: function; getSimSupportCode
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a5ed9cae169e9808c6c8823e624880451:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSimSupportCode() const

Gets support code to be made available within the synapse kernel/function.

This is intended to contain user defined device functions that are used in the weight update code. Preprocessor defines are also allowed if appropriately safeguarded against multiple definition by using ifndef; functions should be declared as "\__host\__ \__device\__" to be available for both GPU and CPU versions; note that this support code is available to sim, event threshold and event code

.. index:: pair: function; getLearnPostSupportCode
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1ac5d1d2d7524cab0f19e965159dd58e8b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getLearnPostSupportCode() const

Gets support code to be made available within learnSynapsesPost kernel/function.

Preprocessor defines are also allowed if appropriately safeguarded against multiple definition by using ifndef; functions should be declared as "\__host\__ \__device\__" to be available for both GPU and CPU versions.

.. index:: pair: function; getSynapseDynamicsSuppportCode
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1aca2d11a28a6cb587dba5f7ae9c87c445:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSynapseDynamicsSuppportCode() const

Gets support code to be made available within the synapse dynamics kernel/function.

Preprocessor defines are also allowed if appropriately safeguarded against multiple definition by using ifndef; functions should be declared as "\__host\__ \__device\__" to be available for both GPU and CPU versions.

.. index:: pair: function; getPreSpikeCode
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a09e5ecd955d9a89bb8deeb5858fa718a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getPreSpikeCode() const

Gets code to be run once per spiking presynaptic neuron before sim code is run on synapses

This is typically for the code to update presynaptic variables. Postsynaptic and synapse variables are not accesible from within this code

.. index:: pair: function; getPostSpikeCode
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a2eab2ca9adfa8698ffe90392b41d1435:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getPostSpikeCode() const

Gets code to be run once per spiking postsynaptic neuron before learn post code is run on synapses

This is typically for the code to update postsynaptic variables. Presynaptic and synapse variables are not accesible from within this code

.. index:: pair: function; getPreVars
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a59c2e29f7c607d87d9342ee88153013d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getPreVars() const

Gets names and types (as strings) of state variables that are common across all synapses coming from the same presynaptic neuron

.. index:: pair: function; getPostVars
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a9d81ca1fb2686a808e975f974ec4884d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getPostVars() const

Gets names and types (as strings) of state variables that are common across all synapses going to the same postsynaptic neuron

.. index:: pair: function; isPreSpikeTimeRequired
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a26c3071dfdf87eaddb857a535894bf7a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool isPreSpikeTimeRequired() const

Whether presynaptic spike times are needed or not.

.. index:: pair: function; isPostSpikeTimeRequired
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a02fb269c52929c962bab49d86d2ca45e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool isPostSpikeTimeRequired() const

Whether postsynaptic spike times are needed or not.

.. index:: pair: function; getPreVarIndex
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1add432f1a452d82183e0574d1fe171f75:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t getPreVarIndex(const std::string& varName) const

Find the index of a named presynaptic variable.

.. index:: pair: function; getPostVarIndex
.. _doxid-d2/d05/classWeightUpdateModels_1_1Base_1a4bca317ba20ee97433d03930081deac3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t getPostVarIndex(const std::string& varName) const

Find the index of a named postsynaptic variable.

