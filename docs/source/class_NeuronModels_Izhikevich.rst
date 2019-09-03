.. index:: pair: class; NeuronModels::Izhikevich
.. _doxid-d7/d0a/classNeuronModels_1_1Izhikevich:

class NeuronModels::Izhikevich
==============================

.. toctree::
	:hidden:

Overview
~~~~~~~~

:ref:`Izhikevich <doxid-d7/d0a/classNeuronModels_1_1Izhikevich>` neuron with fixed parameters :ref:`izhikevich2003simple <doxid-d0/de3/citelist_1CITEREF_izhikevich2003simple>`. :ref:`More...<details-d7/d0a/classNeuronModels_1_1Izhikevich>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <neuronModels.h>
	
	class Izhikevich: public :ref:`NeuronModels::Base<doxid-d7/dad/classNeuronModels_1_1Base>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<4> :target:`ParamValues<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1ae94c10cb33862aef367793f56cdbddd8>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<2> :target:`VarValues<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1aed21c7cc4971268506c9114d17e6bf22>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a59bb450aa8a6e840d9445d52095c1486>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a674301f3dc07282505a2350217791ee0>`;

		// methods
	
		static const NeuronModels::Izhikevich* :target:`getInstance<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1acf0e3e311aab815bc4d7bdf8a048239f>`();
		virtual std::string :ref:`getSimCode<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a67669e3ef6477d221b6023c19a357960>`() const;
		virtual std::string :ref:`getThresholdConditionCode<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1af85c7d6485949e74cc766fd5860f55bc>`() const;
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a3a5d58236a09163fce623add3b1095c0>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a4da70b0f1e24f4634b60d15e2e1c6343>`() const;
	};

	// direct descendants

	class :ref:`IzhikevichVariable<doxid-dc/d87/classNeuronModels_1_1IzhikevichVariable>`;

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
		virtual std::string :ref:`getSimCode<doxid-d7/dad/classNeuronModels_1_1Base_1a3de4c7ff580f63c5b0ec12cb461ebd3a>`() const;
		virtual std::string :ref:`getThresholdConditionCode<doxid-d7/dad/classNeuronModels_1_1Base_1a00ffe96ee864dc67936ce75592c6b198>`() const;
		virtual std::string :ref:`getResetCode<doxid-d7/dad/classNeuronModels_1_1Base_1a4bdc01f203f92c2da4d3b1b48109975d>`() const;
		virtual std::string :ref:`getSupportCode<doxid-d7/dad/classNeuronModels_1_1Base_1ada27dc79296ef8368ac2c7ab20ca8c8e>`() const;
		virtual :ref:`Models::Base::ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>` :ref:`getAdditionalInputVars<doxid-d7/dad/classNeuronModels_1_1Base_1afef62c84373334fe4656a754dbb661c7>`() const;
		virtual bool :ref:`isAutoRefractoryRequired<doxid-d7/dad/classNeuronModels_1_1Base_1a32c9b73420bbdf11a373faa4e0cceb09>`() const;

.. _details-d7/d0a/classNeuronModels_1_1Izhikevich:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`Izhikevich <doxid-d7/d0a/classNeuronModels_1_1Izhikevich>` neuron with fixed parameters :ref:`izhikevich2003simple <doxid-d0/de3/citelist_1CITEREF_izhikevich2003simple>`.

It is usually described as

.. math::

	\begin{eqnarray*} \frac{dV}{dt} &=& 0.04 V^2 + 5 V + 140 - U + I, \\ \frac{dU}{dt} &=& a (bV-U), \end{eqnarray*}

I is an external input current and the voltage V is reset to parameter c and U incremented by parameter d, whenever V >= 30 mV. This is paired with a particular integration procedure of two 0.5 ms Euler time steps for the V equation followed by one 1 ms time step of the U equation. Because of its popularity we provide this model in this form here event though due to the details of the usual implementation it is strictly speaking inconsistent with the displayed equations.

Variables are:

* ``V`` - Membrane potential

* ``U`` - Membrane recovery variable

Parameters are:

* ``a`` - time scale of U

* ``b`` - sensitivity of U

* ``c`` - after-spike reset value of V

* ``d`` - after-spike reset value of U

Methods
-------

.. index:: pair: function; getSimCode
.. _doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a67669e3ef6477d221b6023c19a357960:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSimCode() const

Gets the code that defines the execution of one timestep of integration of the neuron model.

The code will refer to  for the value of the variable with name "NN". It needs to refer to the predefined variable "ISYN", i.e. contain , if it is to receive input.

.. index:: pair: function; getThresholdConditionCode
.. _doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1af85c7d6485949e74cc766fd5860f55bc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getThresholdConditionCode() const

Gets code which defines the condition for a true spike in the described neuron model.

This evaluates to a bool (e.g. "V > 20").

.. index:: pair: function; getParamNames
.. _doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a3a5d58236a09163fce623add3b1095c0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.

.. index:: pair: function; getVars
.. _doxid-d7/d0a/classNeuronModels_1_1Izhikevich_1a4da70b0f1e24f4634b60d15e2e1c6343:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getVars() const

Gets names and types (as strings) of model variables.

