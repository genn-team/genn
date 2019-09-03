.. index:: pair: class; InitSparseConnectivitySnippet::OneToOne
.. _doxid-d5/dd3/classInitSparseConnectivitySnippet_1_1OneToOne:

class InitSparseConnectivitySnippet::OneToOne
=============================================

.. toctree::
	:hidden:

Initialises connectivity to a 'one-to-one' diagonal matrix.


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <initSparseConnectivitySnippet.h>
	
	class OneToOne: public :ref:`InitSparseConnectivitySnippet::Base<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base>`
	{
	public:
		// methods
	
		:target:`DECLARE_SNIPPET<doxid-d5/dd3/classInitSparseConnectivitySnippet_1_1OneToOne_1a8cdbf3d3faa0645465813674ddd9beed>`(
			InitSparseConnectivitySnippet::OneToOne,
			0
			);
	
		:target:`SET_ROW_BUILD_CODE<doxid-d5/dd3/classInitSparseConnectivitySnippet_1_1OneToOne_1ac1f11dbe726beeccc1f877aea00a1928>`("$(addSynapse, $(id_pre));\n""$(endRow);\n");
		:target:`SET_MAX_ROW_LENGTH<doxid-d5/dd3/classInitSparseConnectivitySnippet_1_1OneToOne_1a8b0dac1b1056ac7aea98d1b108252314>`(1);
		:target:`SET_MAX_COL_LENGTH<doxid-d5/dd3/classInitSparseConnectivitySnippet_1_1OneToOne_1a84265bb1cfe611cacd4c3ceba26215ef>`(1);
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

