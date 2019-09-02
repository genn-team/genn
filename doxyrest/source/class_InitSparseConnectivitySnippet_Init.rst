.. index:: pair: class; InitSparseConnectivitySnippet::Init
.. _doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init:

class InitSparseConnectivitySnippet::Init
=========================================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <initSparseConnectivitySnippet.h>
	
	class Init: public :ref:`Snippet::Init<doxid-d8/df6/classSnippet_1_1Init>`
	{
	public:
		// construction
	
		:target:`Init<doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init_1ae7016585811c7d0889832be64b40c8e9>`(
			const :ref:`Base<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base>`* snippet,
			const std::vector<double>& params
			);
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// methods
	
		const SnippetBase* :ref:`getSnippet<doxid-d8/df6/classSnippet_1_1Init_1acc38860261805f85e85a69661623a0d2>`() const;
		const std::vector<double>& :ref:`getParams<doxid-d8/df6/classSnippet_1_1Init_1a82d9ffd36d23da70dd3d206f8f54649c>`() const;
		const std::vector<double>& :ref:`getDerivedParams<doxid-d8/df6/classSnippet_1_1Init_1a293e3c5bd9130372a3b0cdf0312f4590>`() const;
		void :ref:`initDerivedParams<doxid-d8/df6/classSnippet_1_1Init_1a567ca2921ef0b3906c8a418aff1124c0>`(double dt);

