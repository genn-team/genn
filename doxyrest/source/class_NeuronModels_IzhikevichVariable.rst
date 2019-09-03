.. index:: pair: class; NeuronModels::IzhikevichVariable
.. _doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable:

class NeuronModels::IzhikevichVariable
======================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

:ref:`Izhikevich <doxid-d7/d0a/classNeuronModels_1_1Izhikevich>` neuron with variable parameters :ref:`izhikevich2003simple <doxid-d0/de3/citelist_1CITEREF_izhikevich2003simple>`. :ref:`More...<details-dc/d87/classNeuronModels_1_1IzhikevichVariable>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <neuronModels.h>
	
	class IzhikevichVariable: public :ref:`NeuronModels::Izhikevich<doxid-d7/d0a/classNeuronModels_1_1Izhikevich>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<0> :target:`ParamValues<doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable_1a96e038918c5db3058040d60e6e67c400>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<6> :target:`VarValues<doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable_1adbf8876ce890d29615b5f4e3bd9ac46a>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable_1a7a76ca4abb9214d8da22bd2987906e26>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable_1aa35945b830120db0dddbc465f1e724fa>`;

		// methods
	
		static const NeuronModels::IzhikevichVariable* :target:`getInstance<doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable_1a69d5c6810f1bdedd118b956dfca7c7f6>`();
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable_1a77955c438418e0c215e31a527823c689>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable_1a8c4a88ba3d44ca550325148203bc53ca>`() const;
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
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<4> :ref:`ParamValues<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1ae94c10cb33862aef367793f56cdbddd8>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<2> :ref:`VarValues<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1aed21c7cc4971268506c9114d17e6bf22>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :ref:`PreVarValues<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a59bb450aa8a6e840d9445d52095c1486>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :ref:`PostVarValues<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a674301f3dc07282505a2350217791ee0>`;

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
		static const :ref:`NeuronModels::Izhikevich<doxid-d7/d0a/classNeuronModels_1_1Izhikevich>`* :ref:`getInstance<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1acf0e3e311aab815bc4d7bdf8a048239f>`();
		virtual std::string :ref:`getSimCode<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a67669e3ef6477d221b6023c19a357960>`() const;
		virtual std::string :ref:`getThresholdConditionCode<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1af85c7d6485949e74cc766fd5860f55bc>`() const;
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a3a5d58236a09163fce623add3b1095c0>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a4da70b0f1e24f4634b60d15e2e1c6343>`() const;

.. _details-dc/d87/classNeuronModels_1_1IzhikevichVariable:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`Izhikevich <doxid-d7/d0a/classNeuronModels_1_1Izhikevich>` neuron with variable parameters :ref:`izhikevich2003simple <doxid-d0/de3/citelist_1CITEREF_izhikevich2003simple>`.

This is the same model as :ref:`Izhikevich <doxid-d7/d0a/classNeuronModels_1_1Izhikevich>` but parameters are defined as "variables" in order to allow users to provide individual values for each individual neuron instead of fixed values for all neurons across the population.

Accordingly, the model has the Variables:

* ``V`` - Membrane potential

* ``U`` - Membrane recovery variable

* ``a`` - time scale of U

* ``b`` - sensitivity of U

* ``c`` - after-spike reset value of V

* ``d`` - after-spike reset value of U

and no parameters.

Methods
-------

.. index:: pair: function; getParamNames
.. _doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable_1a77955c438418e0c215e31a527823c689:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.

.. index:: pair: function; getVars
.. _doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable_1a8c4a88ba3d44ca550325148203bc53ca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getVars() const

Gets names and types (as strings) of model variables.

