.. index:: pair: class; NeuronModels::RulkovMap
.. _doxid-db/d23/classNeuronModels_1_1RulkovMap:

class NeuronModels::RulkovMap
=============================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Rulkov Map neuron. :ref:`More...<details-db/d23/classNeuronModels_1_1RulkovMap>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <neuronModels.h>
	
	class RulkovMap: public :ref:`NeuronModels::Base<doxid-d7/dad/classNeuronModels_1_1Base>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<4> :target:`ParamValues<doxid-db/d23/classNeuronModels_1_1RulkovMap_1ab78959f43ae5b4e46c9a0be793f4bacc>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<2> :target:`VarValues<doxid-db/d23/classNeuronModels_1_1RulkovMap_1accfe1383d9c7816d0b9163d28a8edd19>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-db/d23/classNeuronModels_1_1RulkovMap_1a9f4f6a5071fe21306d101838e04fd220>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-db/d23/classNeuronModels_1_1RulkovMap_1a29e31ae2d2fbfe736f7e05f9f4c5bfc1>`;

		// methods
	
		static const NeuronModels::RulkovMap* :target:`getInstance<doxid-db/d23/classNeuronModels_1_1RulkovMap_1aa738f74b47ff1be4dc9c9d21cbd77906>`();
		virtual std::string :ref:`getSimCode<doxid-db/d23/classNeuronModels_1_1RulkovMap_1a54bf2e930d391bd5989845beb75b0757>`() const;
		virtual std::string :ref:`getThresholdConditionCode<doxid-db/d23/classNeuronModels_1_1RulkovMap_1a35d4a3c3421f16e513c463d74132c2ad>`() const;
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d23/classNeuronModels_1_1RulkovMap_1abb0d6b2673bd56d4f11822e4b56b9342>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-db/d23/classNeuronModels_1_1RulkovMap_1a4b38f779fbd61198a040d1aba3b17159>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d23/classNeuronModels_1_1RulkovMap_1aa505830a548e3e112da9bd8cf09aec0e>`() const;
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

.. _details-db/d23/classNeuronModels_1_1RulkovMap:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Rulkov Map neuron.

The :ref:`RulkovMap <doxid-db/d23/classNeuronModels_1_1RulkovMap>` type is a map based neuron model based on :ref:`[5] <doxid-d0/de3/citelist_1CITEREF_Rulkov2002>` but in the 1-dimensional map form used in :ref:`[3] <doxid-d0/de3/citelist_1CITEREF_nowotny2005self>` :

.. math::

	\begin{eqnarray*} V(t+\Delta t) &=& \left\{ \begin{array}{ll} V_{\rm spike} \Big(\frac{\alpha V_{\rm spike}}{V_{\rm spike}-V(t) \beta I_{\rm syn}} + y \Big) & V(t) \leq 0 \\ V_{\rm spike} \big(\alpha+y\big) & V(t) \leq V_{\rm spike} \big(\alpha + y\big) \; \& \; V(t-\Delta t) \leq 0 \\ -V_{\rm spike} & {\rm otherwise} \end{array} \right. \end{eqnarray*}

The ``:ref:`RulkovMap <doxid-db/d23/classNeuronModels_1_1RulkovMap>``` type only works as intended for the single time step size of ``DT`` = 0.5.

The ``:ref:`RulkovMap <doxid-db/d23/classNeuronModels_1_1RulkovMap>``` type has 2 variables:

* ``V`` - the membrane potential

* ``preV`` - the membrane potential at the previous time step

and it has 4 parameters:

* ``Vspike`` - determines the amplitude of spikes, typically -60mV

* ``alpha`` - determines the shape of the iteration function, typically :math:`\alpha` = 3

* ``y`` - "shift / excitation" parameter, also determines the iteration function,originally, y= -2.468

* ``beta`` - roughly speaking equivalent to the input resistance, i.e. it regulates the scale of the input into the neuron, typically :math:`\beta` = 2.64 :math:`{\rm M}\Omega`.

The initial values array for the ``:ref:`RulkovMap <doxid-db/d23/classNeuronModels_1_1RulkovMap>``` type needs two entries for ``V`` and ``Vpre`` and the parameter array needs four entries for ``Vspike``, ``alpha``, ``y`` and ``beta``, *in that order*.

Methods
-------

.. index:: pair: function; getSimCode
.. _doxid-db/d23/classNeuronModels_1_1RulkovMap_1a54bf2e930d391bd5989845beb75b0757:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSimCode() const

Gets the code that defines the execution of one timestep of integration of the neuron model.

The code will refer to  for the value of the variable with name "NN". It needs to refer to the predefined variable "ISYN", i.e. contain , if it is to receive input.

.. index:: pair: function; getThresholdConditionCode
.. _doxid-db/d23/classNeuronModels_1_1RulkovMap_1a35d4a3c3421f16e513c463d74132c2ad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getThresholdConditionCode() const

Gets code which defines the condition for a true spike in the described neuron model.

This evaluates to a bool (e.g. "V > 20").

.. index:: pair: function; getParamNames
.. _doxid-db/d23/classNeuronModels_1_1RulkovMap_1abb0d6b2673bd56d4f11822e4b56b9342:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.

.. index:: pair: function; getVars
.. _doxid-db/d23/classNeuronModels_1_1RulkovMap_1a4b38f779fbd61198a040d1aba3b17159:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getVars() const

Gets names and types (as strings) of model variables.

.. index:: pair: function; getDerivedParams
.. _doxid-db/d23/classNeuronModels_1_1RulkovMap_1aa505830a548e3e112da9bd8cf09aec0e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` getDerivedParams() const

Gets names of derived model parameters and the function objects to call to Calculate their value from a vector of model parameter values

