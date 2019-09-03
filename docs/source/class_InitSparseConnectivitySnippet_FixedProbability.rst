.. index:: pair: class; InitSparseConnectivitySnippet::FixedProbability
.. _doxid-df/d2e/classInitSparseConnectivitySnippet_1_1FixedProbability:

class InitSparseConnectivitySnippet::FixedProbability
=====================================================

.. toctree::
	:hidden:

Initialises connectivity with a fixed probability of a synapse existing between a pair of pre and postsynaptic neurons.

Whether a synapse exists between a pair of pre and a postsynaptic neurons can be modelled using a Bernoulli distribution. While this COULD br sampling directly by repeatedly drawing from the uniform distribution, this is innefficient. Instead we sample from the gemetric distribution which describes "the probability distribution of the number of Bernoulli
trials needed to get one success" essentially the distribution of the 'gaps' between synapses. We do this using the "inversion method" described by Devroye (1986) essentially inverting the CDF of the equivalent continuous distribution (in this case the exponential distribution)


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <initSparseConnectivitySnippet.h>
	
	class FixedProbability: public :ref:`InitSparseConnectivitySnippet::FixedProbabilityBase<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase>`
	{
	public:
		// methods
	
		:target:`DECLARE_SNIPPET<doxid-df/d2e/classInitSparseConnectivitySnippet_1_1FixedProbability_1ad89b5fc9427cffbcca1624a0186b520d>`(
			InitSparseConnectivitySnippet::FixedProbability,
			1
			);
	
		:target:`SET_ROW_BUILD_CODE<doxid-df/d2e/classInitSparseConnectivitySnippet_1_1FixedProbability_1ac9c0c075e34997dd9f78f28e6c18c86a>`();
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
		typedef std::function<unsigned int(unsigned int, unsigned int, const std::vector<double>&)> :ref:`CalcMaxLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7719c85cf11d180023fa955ec86a4204>`;

		// structs
	
		struct :ref:`DerivedParam<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam>`;
		struct :ref:`EGP<doxid-dd/d5d/structSnippet_1_1Base_1_1EGP>`;
		struct :ref:`ParamVal<doxid-d7/dda/structSnippet_1_1Base_1_1ParamVal>`;

		// methods
	
		virtual :ref:`~Base<doxid-db/d97/classSnippet_1_1Base_1a17a9ca158277401f2c190afb1e791d1f>`();
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1a0c8374854fbdc457bf0f75e458748580>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1ab01de002618efa59541c927ffdd463f5>`() const;
		virtual std::string :ref:`getRowBuildCode<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1aa9dd29bd22e2a8f369e1f058e8d37d62>`() const;
		virtual :ref:`ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>` :ref:`getRowBuildStateVars<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a2e1599a8871e7ffa6ee63d2da640b4a7>`() const;
		virtual :ref:`CalcMaxLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7719c85cf11d180023fa955ec86a4204>` :ref:`getCalcMaxRowLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1ab164352b017276ef6957ac033a4e70ec>`() const;
		virtual :ref:`CalcMaxLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7719c85cf11d180023fa955ec86a4204>` :ref:`getCalcMaxColLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a43072eecc2ae8b953a6fff561c83c449>`() const;
		virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` :ref:`getExtraGlobalParams<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1ac00e552fb74f8f6fd96939abee7f9f92>`() const;
		size_t :ref:`getExtraGlobalParamIndex<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a051b7e5128dc95bc2151c9f1ae0a2d25>`(const std::string& paramName) const;
		virtual std::string :ref:`getRowBuildCode<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a57f8d74079f8aa82fa90b11f93dce309>`() const = 0;
		:ref:`SET_ROW_BUILD_STATE_VARS<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a98582cf471523527d7c55c7f0351fec1>`({{"prevJ","int",-1}});
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a74482a683eaf4407369df75b28957905>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a8ad49047e343c93b0c92be50b57ae7f5>`() const;
		:ref:`SET_CALC_MAX_ROW_LENGTH_FUNC<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a14975e365e173dc24134c26ac8cf5a91>`((unsigned int numPre, unsigned int numPost, const std::vector<double>&pars){const double quantile=pow(0.9999, 1.0/(double) numPre);return :ref:`binomialInverseCDF<doxid-d6/d24/binomial_8cc_1a620a939ae672f5750398dcfa48e288be>`(quantile, numPost, pars[0]);});
		:ref:`SET_CALC_MAX_COL_LENGTH_FUNC<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1abfc1212c5f01c43426939ca7884161e8>`((unsigned int numPre, unsigned int numPost, const std::vector<double>&pars){const double quantile=pow(0.9999, 1.0/(double) numPost);return :ref:`binomialInverseCDF<doxid-d6/d24/binomial_8cc_1a620a939ae672f5750398dcfa48e288be>`(quantile, numPre, pars[0]);});

