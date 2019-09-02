.. index:: pair: struct; Snippet::Base::DerivedParam
.. _doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam:

struct Snippet::Base::DerivedParam
==================================

.. toctree::
	:hidden:

A derived parameter has a name and a function for obtaining its value.


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <snippet.h>
	
	struct DerivedParam
	{
		// fields
	
		std::string :target:`name<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam_1a896416e06e3eded953331be4780abcc3>`;
		std::function<double(const std::vector<double>&, double)> :target:`func<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam_1a4b29bbe2e353521b8499128a73ec5a2b>`;
	};
