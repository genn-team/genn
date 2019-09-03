.. index:: pair: class; NeuronModels::TraubMilesAlt
.. _doxid-d0/df2/classNeuronModels_1_1TraubMilesAlt:

class NeuronModels::TraubMilesAlt
=================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Hodgkin-Huxley neurons with Traub & Miles algorithm. :ref:`More...<details-d0/df2/classNeuronModels_1_1TraubMilesAlt>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <neuronModels.h>
	
	class TraubMilesAlt: public :ref:`NeuronModels::TraubMiles<doxid-d2/dc3/classNeuronModels_1_1TraubMiles>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<7> :target:`ParamValues<doxid-d0/df2/classNeuronModels_1_1TraubMilesAlt_1a5d71461d6e76cc357c4e48df364c5478>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<4> :target:`VarValues<doxid-d0/df2/classNeuronModels_1_1TraubMilesAlt_1a3da9ad33bc11ee84dee6853bd14776d1>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-d0/df2/classNeuronModels_1_1TraubMilesAlt_1aa54e3356d02d600123b3ae4da9d27db5>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-d0/df2/classNeuronModels_1_1TraubMilesAlt_1acd5c2471f44cf245d7d7807ed4cb7c49>`;

		// methods
	
		static const NeuronModels::TraubMilesAlt* :target:`getInstance<doxid-d0/df2/classNeuronModels_1_1TraubMilesAlt_1aaba12e49ac76b945a8678fc03a44a914>`();
		virtual std::string :ref:`getSimCode<doxid-d0/df2/classNeuronModels_1_1TraubMilesAlt_1a880d0af5b1bb49eecfd0bee29716a8aa>`() const;
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
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<7> :ref:`ParamValues<doxid-d2/dc3/classNeuronModels_1_1TraubMiles_1a968f4e8ff125d312cf760d36f0dee95e>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<4> :ref:`VarValues<doxid-d2/dc3/classNeuronModels_1_1TraubMiles_1a8ec817cdd9f7bbf5432ffca45dcb5fd2>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :ref:`PreVarValues<doxid-d2/dc3/classNeuronModels_1_1TraubMiles_1a8d8b78952c1df2ada32ce9ba9e89dba2>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :ref:`PostVarValues<doxid-d2/dc3/classNeuronModels_1_1TraubMiles_1a4a2c785c6a7b36aa64026e032c0eeaea>`;

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
		virtual std::string :ref:`getSimCode<doxid-d7/dad/classNeuronModels_1_1Base_1a3de4c7ff580f63c5b0ec12cb461ebd3a>`() const;
		virtual std::string :ref:`getThresholdConditionCode<doxid-d7/dad/classNeuronModels_1_1Base_1a00ffe96ee864dc67936ce75592c6b198>`() const;
		virtual std::string :ref:`getResetCode<doxid-d7/dad/classNeuronModels_1_1Base_1a4bdc01f203f92c2da4d3b1b48109975d>`() const;
		virtual std::string :ref:`getSupportCode<doxid-d7/dad/classNeuronModels_1_1Base_1ada27dc79296ef8368ac2c7ab20ca8c8e>`() const;
		virtual :ref:`Models::Base::ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>` :ref:`getAdditionalInputVars<doxid-d7/dad/classNeuronModels_1_1Base_1afef62c84373334fe4656a754dbb661c7>`() const;
		virtual bool :ref:`isAutoRefractoryRequired<doxid-d7/dad/classNeuronModels_1_1Base_1a32c9b73420bbdf11a373faa4e0cceb09>`() const;
		static const :ref:`NeuronModels::TraubMiles<doxid-d2/dc3/classNeuronModels_1_1TraubMiles>`* :ref:`getInstance<doxid-d2/dc3/classNeuronModels_1_1TraubMiles_1ae47f72166dbcb1e55d2e07755b1526da>`();
		virtual std::string :ref:`getSimCode<doxid-d2/dc3/classNeuronModels_1_1TraubMiles_1a868f03fe6e30b947586cdb6a8d29146a>`() const;
		virtual std::string :ref:`getThresholdConditionCode<doxid-d2/dc3/classNeuronModels_1_1TraubMiles_1afdc035bc1c4f29c57c589b05678d232f>`() const;
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-d2/dc3/classNeuronModels_1_1TraubMiles_1a248a2c830b67bb80cb6dbbbda57f405d>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d2/dc3/classNeuronModels_1_1TraubMiles_1a9c141ce37428493b4fdfb0f55e63d303>`() const;

.. _details-d0/df2/classNeuronModels_1_1TraubMilesAlt:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Hodgkin-Huxley neurons with Traub & Miles algorithm.

Using a workaround to avoid singularity: adding the munimum numerical value of the floating point precision used.

Methods
-------

.. index:: pair: function; getSimCode
.. _doxid-d0/df2/classNeuronModels_1_1TraubMilesAlt_1a880d0af5b1bb49eecfd0bee29716a8aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSimCode() const

Gets the code that defines the execution of one timestep of integration of the neuron model.

The code will refer to  for the value of the variable with name "NN". It needs to refer to the predefined variable "ISYN", i.e. contain , if it is to receive input.

