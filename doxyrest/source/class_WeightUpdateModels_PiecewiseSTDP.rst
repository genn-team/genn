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

.. _details-df/d86/classWeightUpdateModels_1_1PiecewiseSTDP:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

This is a simple STDP rule including a time delay for the finite transmission speed of the synapse.

The STDP window is defined as a piecewise function:

.. image:: LEARN1SYNAPSE_explain_html.png



.. image:: LEARN1SYNAPSE_explain.png
	:width: 10

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

