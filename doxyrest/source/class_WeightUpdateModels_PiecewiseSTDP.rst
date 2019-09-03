.. index:: pair: class; WeightUpdateModels::PiecewiseSTDP
.. _doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP:

class WeightUpdateModels::PiecewiseSTDP
=======================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

This is a simple STDP rule including a time delay for the finite transmission speed of the synapse. :ref:`More...<details-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <weightUpdateModels.h>
	
	class PiecewiseSTDP: public :ref:`WeightUpdateModels::Base<doxid-d2/d05/classWeightUpdateModels_1_1Base>`
	{
	public:
		// methods
	
		:target:`DECLARE_WEIGHT_UPDATE_MODEL<doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a9ef28608d86ff2be8e68e9a97134ce05>`(
			PiecewiseSTDP,
			10,
			2,
			0,
			0
			);
	
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a3401601af479387c1dcbc4d986741c81>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1aef7011a4e9e3975ee432e2f901ccf4a9>`() const;
		virtual std::string :ref:`getSimCode<doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1aaf6969f0f388155f2efedca6015c0b70>`() const;
		virtual std::string :ref:`getLearnPostCode<doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a8e054d3dd776a3b8d07af81c5d7d8647>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a9e85e5c8178425856929c747d53c421f>`() const;
		virtual bool :ref:`isPreSpikeTimeRequired<doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a060eb8fe3f26e106c662519d23a93717>`() const;
		virtual bool :ref:`isPostSpikeTimeRequired<doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a03875502d4f8594bfd1f53d248ae63c2>`() const;
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

.. _details-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

This is a simple STDP rule including a time delay for the finite transmission speed of the synapse.

The STDP window is defined as a piecewise function:

.. image:: LEARN1SYNAPSE_explain_html.png



.. image:: LEARN1SYNAPSE_explain.png
	:alt: width=10cm

The STDP curve is applied to the raw synaptic conductance ``gRaw``, which is then filtered through the sugmoidal filter displayed above to obtain the value of ``g``.

The STDP curve implies that unpaired pre- and post-synaptic spikes incur a negative increment in ``gRaw`` (and hence in ``g``).

The time of the last spike in each neuron, "sTXX", where XX is the name of a neuron population is (somewhat arbitrarily) initialised to -10.0 ms. If neurons never spike, these spike times are used.

It is the raw synaptic conductance ``gRaw`` that is subject to the STDP rule. The resulting synaptic conductance is a sigmoid filter of ``gRaw``. This implies that ``g`` is initialised but not ``gRaw``, the synapse will revert to the value that corresponds to ``gRaw``.

An example how to use this synapse correctly is given in ``map_classol.cc`` (MBody1 userproject):

.. ref-code-block:: cpp

	for (int i= 0; i < model.neuronN[1]*model.neuronN[3]; i++) {
	        if (gKCDN[i] < 2.0*SCALAR_MIN){
	            cnt++;
	            fprintf(stdout, "Too low conductance value %e detected and set to 2*SCALAR_MIN= %e, at index %d \n", gKCDN[i], 2*SCALAR_MIN, i);
	            gKCDN[i] = 2.0*SCALAR_MIN; //to avoid log(0)/0 below
	        }
	        scalar tmp = gKCDN[i] / myKCDN_p[5]*2.0 ;
	        gRawKCDN[i]=  0.5 * log( tmp / (2.0 - tmp)) /myKCDN_p[7] + myKCDN_p[6];
	}
	cerr << "Total number of low value corrections: " << cnt << endl;

One cannot set values of ``g`` fully to ``0``, as this leads to ``gRaw`` = -infinity and this is not support. I.e., 'g' needs to be some nominal value > 0 (but can be extremely small so that it acts like it's 0).

The model has 2 variables:

* ``g:`` conductance of ``scalar`` type

* ``gRaw:`` raw conductance of ``scalar`` type

Parameters are (compare to the figure above):

* ``tLrn:`` Time scale of learning changes

* ``tChng:`` Width of learning window

* ``tDecay:`` Time scale of synaptic strength decay

* ``tPunish10:`` Time window of suppression in response to 1/0

* ``tPunish01:`` Time window of suppression in response to 0/1

* ``gMax:`` Maximal conductance achievable

* ``gMid:`` Midpoint of sigmoid g filter curve

* ``gSlope:`` Slope of sigmoid g filter curve

* ``tauShift:`` Shift of learning curve

* ``gSyn0:`` Value of syn conductance g decays to

Methods
-------

.. index:: pair: function; getParamNames
.. _doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a3401601af479387c1dcbc4d986741c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.

.. index:: pair: function; getVars
.. _doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1aef7011a4e9e3975ee432e2f901ccf4a9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getVars() const

Gets names and types (as strings) of model variables.

.. index:: pair: function; getSimCode
.. _doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1aaf6969f0f388155f2efedca6015c0b70:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getSimCode() const

Gets simulation code run when 'true' spikes are received.

.. index:: pair: function; getLearnPostCode
.. _doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a8e054d3dd776a3b8d07af81c5d7d8647:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getLearnPostCode() const

Gets code to include in the learnSynapsesPost kernel/function.

For examples when modelling STDP, this is where the effect of postsynaptic spikes which occur *after* presynaptic spikes are applied.

.. index:: pair: function; getDerivedParams
.. _doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a9e85e5c8178425856929c747d53c421f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` getDerivedParams() const

Gets names of derived model parameters and the function objects to call to Calculate their value from a vector of model parameter values

.. index:: pair: function; isPreSpikeTimeRequired
.. _doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a060eb8fe3f26e106c662519d23a93717:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool isPreSpikeTimeRequired() const

Whether presynaptic spike times are needed or not.

.. index:: pair: function; isPostSpikeTimeRequired
.. _doxid-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP_1a03875502d4f8594bfd1f53d248ae63c2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool isPostSpikeTimeRequired() const

Whether postsynaptic spike times are needed or not.

