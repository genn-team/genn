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
		// construction
	
		:target:`Init<doxid-d8/df6/classSnippet_1_1Init_1ae1b490fba08a926f30d03977b37339c5>`(
			const SnippetBase* snippet,
			const std::vector<double>& params
			);

		// methods
	
		const SnippetBase* :target:`getSnippet<doxid-d8/df6/classSnippet_1_1Init_1acc38860261805f85e85a69661623a0d2>`() const;
		const std::vector<double>& :target:`getParams<doxid-d8/df6/classSnippet_1_1Init_1a82d9ffd36d23da70dd3d206f8f54649c>`() const;
		const std::vector<double>& :target:`getDerivedParams<doxid-d8/df6/classSnippet_1_1Init_1a293e3c5bd9130372a3b0cdf0312f4590>`() const;
		void :target:`initDerivedParams<doxid-d8/df6/classSnippet_1_1Init_1a567ca2921ef0b3906c8a418aff1124c0>`(double dt);
	};

	// direct descendants

	class :ref:`Init<doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init>`;
	class :ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`;
