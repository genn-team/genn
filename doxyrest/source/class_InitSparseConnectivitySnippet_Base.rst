.. index:: pair: class; InitSparseConnectivitySnippet::Base
.. _doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base:

class InitSparseConnectivitySnippet::Base
=========================================

.. toctree::
	:hidden:

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <initSparseConnectivitySnippet.h>
	
	class Base: public :ref:`Snippet::Base<doxid-db/d97/classSnippet_1_1Base>`
	{
	public:
		// typedefs
	
		typedef std::function<unsigned int(unsigned int, unsigned int, const std::vector<double>&)> :target:`CalcMaxLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7719c85cf11d180023fa955ec86a4204>`;

		// methods
	
		virtual std::string :target:`getRowBuildCode<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1aa9dd29bd22e2a8f369e1f058e8d37d62>`() const;
		virtual :ref:`ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>` :target:`getRowBuildStateVars<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a2e1599a8871e7ffa6ee63d2da640b4a7>`() const;
		virtual :ref:`CalcMaxLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7719c85cf11d180023fa955ec86a4204>` :ref:`getCalcMaxRowLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1ab164352b017276ef6957ac033a4e70ec>`() const;
		virtual :ref:`CalcMaxLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7719c85cf11d180023fa955ec86a4204>` :ref:`getCalcMaxColLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a43072eecc2ae8b953a6fff561c83c449>`() const;
		virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` :ref:`getExtraGlobalParams<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1ac00e552fb74f8f6fd96939abee7f9f92>`() const;
		size_t :ref:`getExtraGlobalParamIndex<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a051b7e5128dc95bc2151c9f1ae0a2d25>`(const std::string& paramName) const;
	};

	// direct descendants

	class :ref:`FixedProbabilityBase<doxid-db/d69/classInitSparseConnectivitySnippet_1_1FixedProbabilityBase>`;
	class :ref:`OneToOne<doxid-d5/dd3/classInitSparseConnectivitySnippet_1_1OneToOne>`;
	class :ref:`Uninitialised<doxid-d1/d54/classInitSparseConnectivitySnippet_1_1Uninitialised>`;

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

		// structs
	
		struct :ref:`DerivedParam<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam>`;
		struct :ref:`EGP<doxid-dd/d5d/structSnippet_1_1Base_1_1EGP>`;
		struct :ref:`ParamVal<doxid-d7/dda/structSnippet_1_1Base_1_1ParamVal>`;

		// methods
	
		virtual :ref:`~Base<doxid-db/d97/classSnippet_1_1Base_1a17a9ca158277401f2c190afb1e791d1f>`();
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1a0c8374854fbdc457bf0f75e458748580>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1ab01de002618efa59541c927ffdd463f5>`() const;

.. _details-d5/d9f/classInitSparseConnectivitySnippet_1_1Base:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Methods
-------

.. index:: pair: function; getCalcMaxRowLengthFunc
.. _doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1ab164352b017276ef6957ac033a4e70ec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`CalcMaxLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7719c85cf11d180023fa955ec86a4204>` getCalcMaxRowLengthFunc() const

Get function to calculate the maximum row length of this connector based on the parameters and the size of the pre and postsynaptic population.

.. index:: pair: function; getCalcMaxColLengthFunc
.. _doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a43072eecc2ae8b953a6fff561c83c449:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`CalcMaxLengthFunc<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a7719c85cf11d180023fa955ec86a4204>` getCalcMaxColLengthFunc() const

Get function to calculate the maximum column length of this connector based on the parameters and the size of the pre and postsynaptic population.

.. index:: pair: function; getExtraGlobalParams
.. _doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1ac00e552fb74f8f6fd96939abee7f9f92:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` getExtraGlobalParams() const

Gets names and types (as strings) of additional per-population parameters for the connection initialisation snippet

.. index:: pair: function; getExtraGlobalParamIndex
.. _doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base_1a051b7e5128dc95bc2151c9f1ae0a2d25:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t getExtraGlobalParamIndex(const std::string& paramName) const

Find the index of a named extra global parameter.

