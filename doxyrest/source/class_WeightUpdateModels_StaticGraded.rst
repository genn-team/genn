.. index:: pair: class; WeightUpdateModels::StaticGraded
.. _doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded:

class WeightUpdateModels::StaticGraded
======================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Graded-potential, static synapse. :ref:`More...<details-d6/d64/classWeightUpdateModels_1_1StaticGraded>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <weightUpdateModels.h>
	
	class StaticGraded: public :ref:`WeightUpdateModels::Base<doxid-d2/d05/classWeightUpdateModels_1_1Base>`
	{
	public:
		// methods
	
		:target:`DECLARE_WEIGHT_UPDATE_MODEL<doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded_1af74719ff8e1d08a83b88372f52ebd833>`(
			StaticGraded,
			2,
			1,
			0,
			0
			);
	
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded_1a275644bd25a6b9446073ac90c0cd46ac>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded_1ae6bbca6784c1a6fc51b8eee7d0362660>`() const;
		virtual std::string :ref:`getEventCode<doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded_1a4358fcaf3d6c2b05705370f5fd601882>`() const;
		virtual std::string :ref:`getEventThresholdConditionCode<doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded_1a403cb623a28d4c89eb1097c48bb6210e>`() const;
	};

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

.. _details-d6/d64/classWeightUpdateModels_1_1StaticGraded:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Graded-potential, static synapse.

In a graded synapse, the conductance is updated gradually with the rule:

.. math::

	gSyn= g * tanh((V - E_{pre}) / V_{slope}

whenever the membrane potential :math:`V` is larger than the threshold :math:`E_{pre}`. The model has 1 variable:

* ``g:`` conductance of ``scalar`` type

The parameters are:

* ``Epre:`` Presynaptic threshold potential

* ``Vslope:`` Activation slope of graded release

``event`` code is:

.. ref-code-block:: cpp

	$(addToInSyn, $(g)* tanh(($(V_pre)-($(Epre)))*DT*2/$(Vslope)));

``event`` threshold condition code is:

.. ref-code-block:: cpp

	$(V_pre) > $(Epre)

The pre-synaptic variables are referenced with the suffix ``_pre`` in synapse related code such as an the event threshold test. Users can also access post-synaptic neuron variables using the suffix ``_post``.

Methods
-------

.. index:: pair: function; getParamNames
.. _doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded_1a275644bd25a6b9446073ac90c0cd46ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.

.. index:: pair: function; getVars
.. _doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded_1ae6bbca6784c1a6fc51b8eee7d0362660:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getVars() const

Gets names and types (as strings) of model variables.

.. index:: pair: function; getEventCode
.. _doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded_1a4358fcaf3d6c2b05705370f5fd601882:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getEventCode() const

Gets code run when events (all the instances where event threshold condition is met) are received.

.. index:: pair: function; getEventThresholdConditionCode
.. _doxid-d6/d64/classWeightUpdateModels_1_1StaticGraded_1a403cb623a28d4c89eb1097c48bb6210e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getEventThresholdConditionCode() const

Gets codes to test for events.

