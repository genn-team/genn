.. index:: pair: class; NeuronModels::PoissonNew
.. _doxid-dc/dc0/classNeuronModels_1_1PoissonNew:

class NeuronModels::PoissonNew
==============================

.. toctree::
	:hidden:

Overview
~~~~~~~~

:ref:`Poisson <doxid-de/d1d/classNeuronModels_1_1Poisson>` neurons. :ref:`More...<details-dc/dc0/classNeuronModels_1_1PoissonNew>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <neuronModels.h>
	
	class PoissonNew: public :ref:`NeuronModels::Base<doxid-d7/dad/classNeuronModels_1_1Base>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<1> :target:`ParamValues<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a9ad6559238cc1d3666f74eed991cbdaf>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<1> :target:`VarValues<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a08eff4c0f7dffa1f2b217021939de595>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a403ea9362701828bade7119ca8767204>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1ab36c7c3db9a0b3751ebabb1c5ccbf063>`;

		// methods
	
		static const NeuronModels::PoissonNew* :target:`getInstance<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a79b0a6d08ae5effb4f1f4b3657e41d10>`();
		virtual std::string :ref:`getSimCode<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a0f06f9a277e7ce434a66220ca16fd2fb>`() const;
		virtual std::string :ref:`getThresholdConditionCode<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a520cf35fdff7dacca888d18eafcc5d3e>`() const;
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1adffa087f5ef9119034afe192b208dd94>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a9f0134631c2af3687274f432b19ad646>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1ab880c13caef654859997b9944412fe65>`() const;
		:target:`SET_NEEDS_AUTO_REFRACTORY<doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a82cca47654bf8275dbb9c8748ecfff9f>`(false);
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
		virtual std::string :ref:`getSimCode<doxid-d7/dad/classNeuronModels_1_1Base_1a86ee36307800205642da8b80b20deb18>`() const;
		virtual std::string :ref:`getThresholdConditionCode<doxid-d7/dad/classNeuronModels_1_1Base_1a3d5e944aa81d6c0573c201980ad0a1a9>`() const;
		virtual std::string :ref:`getResetCode<doxid-d7/dad/classNeuronModels_1_1Base_1a723efc11dd8743ec033ab9b0f8f0ac7e>`() const;
		virtual std::string :ref:`getSupportCode<doxid-d7/dad/classNeuronModels_1_1Base_1aad7ac85a47cc72aeaba4c38fc636bd38>`() const;
		virtual :ref:`Models::Base::ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>` :ref:`getAdditionalInputVars<doxid-d7/dad/classNeuronModels_1_1Base_1a31b37d82e943c8f8e9221f0012411eca>`() const;
		virtual bool :ref:`isAutoRefractoryRequired<doxid-d7/dad/classNeuronModels_1_1Base_1a84965f7dab4de66215fd85f207ebd9e4>`() const;

.. _details-dc/dc0/classNeuronModels_1_1PoissonNew:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`Poisson <doxid-de/d1d/classNeuronModels_1_1Poisson>` neurons.

It has 1 state variable:

* ``timeStepToSpike`` - Number of timesteps to next spike

and 1 parameter:

* ``rate`` - Mean firing rate (Hz)

Internally this samples from the exponential distribution using the C++ 11 <random> library on the CPU and by transforming the uniform distribution, generated using cuRAND, with a natural log on the GPU.

Methods
-------

.. index:: pair: function; getSimCode
.. _doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a0f06f9a277e7ce434a66220ca16fd2fb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSimCode() const

Gets the code that defines the execution of one timestep of integration of the neuron model.

The code will refer to  for the value of the variable with name "NN". It needs to refer to the predefined variable "ISYN", i.e. contain , if it is to receive input.

.. index:: pair: function; getThresholdConditionCode
.. _doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a520cf35fdff7dacca888d18eafcc5d3e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getThresholdConditionCode() const

Gets code which defines the condition for a true spike in the described neuron model.

This evaluates to a bool (e.g. "V > 20").

.. index:: pair: function; getParamNames
.. _doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1adffa087f5ef9119034afe192b208dd94:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.

.. index:: pair: function; getVars
.. _doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1a9f0134631c2af3687274f432b19ad646:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getVars() const

Gets names and types (as strings) of model variables.

.. index:: pair: function; getDerivedParams
.. _doxid-dc/dc0/classNeuronModels_1_1PoissonNew_1ab880c13caef654859997b9944412fe65:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` getDerivedParams() const

Gets names of derived model parameters and the function objects to call to Calculate their value from a vector of model parameter values

