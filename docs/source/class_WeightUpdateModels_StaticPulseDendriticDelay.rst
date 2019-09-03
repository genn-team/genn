.. index:: pair: class; WeightUpdateModels::StaticPulseDendriticDelay
.. _doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay:

class WeightUpdateModels::StaticPulseDendriticDelay
===================================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Pulse-coupled, static synapse with heterogenous dendritic delays. :ref:`More...<details-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <weightUpdateModels.h>
	
	class StaticPulseDendriticDelay: public :ref:`WeightUpdateModels::Base<doxid-d2/d05/classWeightUpdateModels_1_1Base>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<0> :target:`ParamValues<doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay_1aac18fd6b85313a3e760eb67bd70eab95>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<2> :target:`VarValues<doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay_1a03a9e1b66239db7eec37a832cc60e55b>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay_1a4261ca4ccd5ff949fb9118e814d926af>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay_1a05f0bb892477d28b33c974e40af5423a>`;

		// methods
	
		static const StaticPulseDendriticDelay* :target:`getInstance<doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay_1a50aec06e96f1216a92550080cc908f6f>`();
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay_1a46604156e563a877ec76e8f85c65908b>`() const;
		virtual std::string :ref:`getSimCode<doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay_1aea75c74a2f04439e32c9dcf5dde47e8b>`() const;
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
	
		virtual :ref:`~Base<doxid-db/d97/classSnippet_1_1Base_1a17a9ca158277401f2c190afb1e791d1f>`();
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1a0c8374854fbdc457bf0f75e458748580>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1ab01de002618efa59541c927ffdd463f5>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d6/d97/classModels_1_1Base_1a9df8ba9bf6d971a574ed4745f6cf946c>`() const;
		virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` :ref:`getExtraGlobalParams<doxid-d6/d97/classModels_1_1Base_1a7fdddb7d19382736b330ade62c441de1>`() const;
		size_t :ref:`getVarIndex<doxid-d6/d97/classModels_1_1Base_1afa0e39df5002efc76448e180f82825e4>`(const std::string& varName) const;
		size_t :ref:`getExtraGlobalParamIndex<doxid-d6/d97/classModels_1_1Base_1ae046c19ad56dfb2808c5f4d2cc7475fe>`(const std::string& paramName) const;
		virtual std::string :ref:`getSimCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1aff2152fb55b5b0148491ca4eed9291eb>`() const;
		virtual std::string :ref:`getEventCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a090f5529defe517fabf84c543209406f>`() const;
		virtual std::string :ref:`getLearnPostCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1abd6d3ec97fb1da0f5750f71c7afc09b1>`() const;
		virtual std::string :ref:`getSynapseDynamicsCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a307cb4e18479682b74972257c5d28dc5>`() const;
		virtual std::string :ref:`getEventThresholdConditionCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a3157e0e66cdc654be4ef4ad67024f84d>`() const;
		virtual std::string :ref:`getSimSupportCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a948b94c553782e9cc05a59bda014fe26>`() const;
		virtual std::string :ref:`getLearnPostSupportCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1af98ae8f3d545f8d66d0f80662bf5b322>`() const;
		virtual std::string :ref:`getSynapseDynamicsSuppportCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a7aae3190642d0bbe7f3f6fa01021783f>`() const;
		virtual std::string :ref:`getPreSpikeCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a79e16d6c154e21a8ca7e56599cbe553b>`() const;
		virtual std::string :ref:`getPostSpikeCode<doxid-d2/d05/classWeightUpdateModels_1_1Base_1abb81b1a933f13ba2af62c088387e186f>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getPreVars<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a949a9adcbc40d4ae9bbb51b2ec08dff5>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getPostVars<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a97a0a4fb30a66bb629cd88306e659105>`() const;
		virtual bool :ref:`isPreSpikeTimeRequired<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a86fb753d87f35b53d789f96c6189a911>`() const;
		virtual bool :ref:`isPostSpikeTimeRequired<doxid-d2/d05/classWeightUpdateModels_1_1Base_1ad93200ea885e60a88c108db10349edea>`() const;
		size_t :ref:`getPreVarIndex<doxid-d2/d05/classWeightUpdateModels_1_1Base_1ac8b3e37eeb3f0034ebba50ec01c2840e>`(const std::string& varName) const;
		size_t :ref:`getPostVarIndex<doxid-d2/d05/classWeightUpdateModels_1_1Base_1a5812718ff39cc394f1c6242e3d3f0987>`(const std::string& varName) const;

.. _details-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Pulse-coupled, static synapse with heterogenous dendritic delays.

No learning rule is applied to the synapse and for each pre-synaptic spikes, the synaptic conductances are simply added to the postsynaptic input variable. The model has 2 variables:

* g - conductance of scalar type

* d - dendritic delay in timesteps and no other parameters.

``sim`` code is:

.. ref-code-block:: cpp

	" $(addToInSynDelay, $(g), $(d));\n\

Methods
-------

.. index:: pair: function; getVars
.. _doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay_1a46604156e563a877ec76e8f85c65908b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getVars() const

Gets names and types (as strings) of model variables.

.. index:: pair: function; getSimCode
.. _doxid-d2/d53/classWeightUpdateModels_1_1StaticPulseDendriticDelay_1aea75c74a2f04439e32c9dcf5dde47e8b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSimCode() const

Gets simulation code run when 'true' spikes are received.

