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
	
		:target:`SET_ROW_BUILD_CODE<doxid-df/d2e/classInitSparseConnectivitySnippet_1_1FixedProbability_1ab73e967faf37253cd38a69182341ff46>`();
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
	
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1aad4f3bb00c5f29cb9d0e3585db3f4e20>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1a450c7783570d875e19bcd8a88d10bbf6>`() const;
		virtual std::string :ref:`getRowBuildCode<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1ac2bab66afe84325ad9d1752910d08981>`() const;
		virtual :ref:`ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>` :ref:`getRowBuildStateVars<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1af9419211b940a9c51fbd6450747e1fc5>`() const;
		virtual :ref:`CalcMaxLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7719c85cf11d180023fa955ec86a4204>` :ref:`getCalcMaxRowLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7cf7c23440db65582ddac36f903fcd5e>`() const;
		virtual :ref:`CalcMaxLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7719c85cf11d180023fa955ec86a4204>` :ref:`getCalcMaxColLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a0e4f6cb6d90ae7ce97f8c6792c16d32d>`() const;
		virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` :ref:`getExtraGlobalParams<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a13516d89df1a3c1567cd619bf1f1e97b>`() const;
		size_t :ref:`getExtraGlobalParamIndex<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1afa1caa86450b5b930d790b7ddb650869>`(const std::string& paramName) const;
		virtual std::string :ref:`getRowBuildCode<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a57f8d74079f8aa82fa90b11f93dce309>`() const = 0;
		:ref:`SET_ROW_BUILD_STATE_VARS<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a7fa338aeb688c18956197ecb5578bc85>`({{"prevJ", "int", -1}});
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a74482a683eaf4407369df75b28957905>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a8ad49047e343c93b0c92be50b57ae7f5>`() const;
		:ref:`SET_CALC_MAX_ROW_LENGTH_FUNC<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a788d2f299d703c2b8868fea4c7435f00>`((unsigned int numPre, unsigned int numPost, const std::vector<double>&pars) { const double quantile=pow(0.9999, 1.0/(double) numPre);return :ref:`binomialInverseCDF<doxid-d6/d24/binomial_8cc_1a620a939ae672f5750398dcfa48e288be>`(quantile, numPost, pars[0]);});
		:ref:`SET_CALC_MAX_COL_LENGTH_FUNC<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase_1a56d113e066e419d6e24a5ed9f2cc1c0e>`((unsigned int numPre, unsigned int numPost, const std::vector<double>&pars) { const double quantile=pow(0.9999, 1.0/(double) numPost);return :ref:`binomialInverseCDF<doxid-d6/d24/binomial_8cc_1a620a939ae672f5750398dcfa48e288be>`(quantile, numPre, pars[0]);});

