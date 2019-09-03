.. index:: pair: class; Snippet::Init
.. _doxid-d8/df6/classSnippet_1_1Init:

template class Snippet::Init
============================

.. toctree::
	:hidden:

Class used to bind together everything required to utilize a snippet

#. A pointer to a variable initialisation snippet

#. The parameters required to control the variable initialisation snippet


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <snippet.h>
	
	template <typename SnippetBase>
	class Init
	{
	public:
		// methods
	
		:target:`Init<doxid-d8/df6/classSnippet_1_1Init_1ae1b490fba08a926f30d03977b37339c5>`(
			const SnippetBase* snippet,
			const std::vector<double>& params
			);
	
		const SnippetBase* :target:`getSnippet<doxid-d8/df6/classSnippet_1_1Init_1a3ceb867b6da08f8cc093b73d7e255718>`() const;
		const std::vector<double>& :target:`getParams<doxid-d8/df6/classSnippet_1_1Init_1aa9ac5d3132df146c7cc83a92f81c5195>`() const;
		const std::vector<double>& :target:`getDerivedParams<doxid-d8/df6/classSnippet_1_1Init_1a114423c5f6999733ad63b0630aa2afcc>`() const;
		void :target:`initDerivedParams<doxid-d8/df6/classSnippet_1_1Init_1a567ca2921ef0b3906c8a418aff1124c0>`(double dt);
	};

	// direct descendants

	class :ref:`Init<doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init>`;
	class :ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`;
