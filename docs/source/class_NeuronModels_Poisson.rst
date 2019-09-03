.. index:: pair: class; NeuronModels::Poisson
.. _doxid-de/d1d/classNeuronModels_1_1Poisson:

class NeuronModels::Poisson
===========================

.. toctree::
	:hidden:

Overview
~~~~~~~~

:ref:`Poisson <doxid-de/d1d/classNeuronModels_1_1Poisson>` neurons. :ref:`More...<details-de/d1d/classNeuronModels_1_1Poisson>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <neuronModels.h>
	
	class Poisson: public :ref:`NeuronModels::Base<doxid-d7/dad/classNeuronModels_1_1Base>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<4> :target:`ParamValues<doxid-de/d1d/classNeuronModels_1_1Poisson_1af5af1f4ac77c218dd5f9503636629ca7>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<2> :target:`VarValues<doxid-de/d1d/classNeuronModels_1_1Poisson_1ab80672183484b5121175a3d745643d70>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-de/d1d/classNeuronModels_1_1Poisson_1a362d4a49fb69448b34e29229fa244419>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-de/d1d/classNeuronModels_1_1Poisson_1a33fd8bc36ae705af6d8d4fdc528659c6>`;

		// methods
	
		static const NeuronModels::Poisson* :target:`getInstance<doxid-de/d1d/classNeuronModels_1_1Poisson_1ac0e09d234156a43676664146348bd8c5>`();
		virtual std::string :ref:`getSimCode<doxid-de/d1d/classNeuronModels_1_1Poisson_1a6358f81e1eb01e319507a16cd40af331>`() const;
		virtual std::string :ref:`getThresholdConditionCode<doxid-de/d1d/classNeuronModels_1_1Poisson_1a81aa0f5a5a1e956dfa8ba5e182461b31>`() const;
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-de/d1d/classNeuronModels_1_1Poisson_1afc4eae8217dafcb4c513fced5ed91cf9>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-de/d1d/classNeuronModels_1_1Poisson_1a2287ccc9e4e2ed221cb15931d520ba9f>`() const;
		virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` :ref:`getExtraGlobalParams<doxid-de/d1d/classNeuronModels_1_1Poisson_1a0e1fd99dcc61ea6b34119bef979f7d4a>`() const;
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
		virtual std::string :ref:`getSimCode<doxid-d7/dad/classNeuronModels_1_1Base_1a3de4c7ff580f63c5b0ec12cb461ebd3a>`() const;
		virtual std::string :ref:`getThresholdConditionCode<doxid-d7/dad/classNeuronModels_1_1Base_1a00ffe96ee864dc67936ce75592c6b198>`() const;
		virtual std::string :ref:`getResetCode<doxid-d7/dad/classNeuronModels_1_1Base_1a4bdc01f203f92c2da4d3b1b48109975d>`() const;
		virtual std::string :ref:`getSupportCode<doxid-d7/dad/classNeuronModels_1_1Base_1ada27dc79296ef8368ac2c7ab20ca8c8e>`() const;
		virtual :ref:`Models::Base::ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>` :ref:`getAdditionalInputVars<doxid-d7/dad/classNeuronModels_1_1Base_1afef62c84373334fe4656a754dbb661c7>`() const;
		virtual bool :ref:`isAutoRefractoryRequired<doxid-d7/dad/classNeuronModels_1_1Base_1a32c9b73420bbdf11a373faa4e0cceb09>`() const;

.. _details-de/d1d/classNeuronModels_1_1Poisson:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`Poisson <doxid-de/d1d/classNeuronModels_1_1Poisson>` neurons.

:ref:`Poisson <doxid-de/d1d/classNeuronModels_1_1Poisson>` neurons have constant membrane potential (``Vrest``) unless they are activated randomly to the ``Vspike`` value if (t- ``SpikeTime``) > ``trefract``.

It has 2 variables:

* ``V`` - Membrane potential (mV)

* ``SpikeTime`` - Time at which the neuron spiked for the last time (ms)

and 4 parameters:

* ``trefract`` - Refractory period (ms)

* ``tspike`` - duration of spike (ms)

* ``Vspike`` - Membrane potential at spike (mV)

* ``Vrest`` - Membrane potential at rest (mV)

The initial values array for the ``:ref:`Poisson <doxid-de/d1d/classNeuronModels_1_1Poisson>``` type needs two entries for ``V``, and ``SpikeTime`` and the parameter array needs four entries for ``therate``, ``trefract``, ``Vspike`` and ``Vrest``, *in that order*.

This model uses a linear approximation for the probability of firing a spike in a given time step of size ``DT``, i.e. the probability of firing is :math:`\lambda` times ``DT`` : :math:`p = \lambda \Delta t`. This approximation is usually very good, especially for typical, quite small time steps and moderate firing rates. However, it is worth noting that the approximation becomes poor for very high firing rates and large time steps.

Methods
-------

.. index:: pair: function; getSimCode
.. _doxid-de/d1d/classNeuronModels_1_1Poisson_1a6358f81e1eb01e319507a16cd40af331:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSimCode() const

Gets the code that defines the execution of one timestep of integration of the neuron model.

The code will refer to  for the value of the variable with name "NN". It needs to refer to the predefined variable "ISYN", i.e. contain , if it is to receive input.

.. index:: pair: function; getThresholdConditionCode
.. _doxid-de/d1d/classNeuronModels_1_1Poisson_1a81aa0f5a5a1e956dfa8ba5e182461b31:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getThresholdConditionCode() const

Gets code which defines the condition for a true spike in the described neuron model.

This evaluates to a bool (e.g. "V > 20").

.. index:: pair: function; getParamNames
.. _doxid-de/d1d/classNeuronModels_1_1Poisson_1afc4eae8217dafcb4c513fced5ed91cf9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.

.. index:: pair: function; getVars
.. _doxid-de/d1d/classNeuronModels_1_1Poisson_1a2287ccc9e4e2ed221cb15931d520ba9f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getVars() const

Gets names and types (as strings) of model variables.

.. index:: pair: function; getExtraGlobalParams
.. _doxid-de/d1d/classNeuronModels_1_1Poisson_1a0e1fd99dcc61ea6b34119bef979f7d4a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` getExtraGlobalParams() const

Gets names and types (as strings) of additional per-population parameters for the weight update model.

