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
		// methods
	
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
	
		:ref:`Init<doxid-d8/df6/classSnippet_1_1Init_1ae1b490fba08a926f30d03977b37339c5>`(const SnippetBase* snippet, const std::vector<double>& params);
		const SnippetBase* :ref:`getSnippet<doxid-d8/df6/classSnippet_1_1Init_1a3ceb867b6da08f8cc093b73d7e255718>`() const;
		const std::vector<double>& :ref:`getParams<doxid-d8/df6/classSnippet_1_1Init_1aa9ac5d3132df146c7cc83a92f81c5195>`() const;
		const std::vector<double>& :ref:`getDerivedParams<doxid-d8/df6/classSnippet_1_1Init_1a114423c5f6999733ad63b0630aa2afcc>`() const;
		void :ref:`initDerivedParams<doxid-d8/df6/classSnippet_1_1Init_1a567ca2921ef0b3906c8a418aff1124c0>`(double dt);

