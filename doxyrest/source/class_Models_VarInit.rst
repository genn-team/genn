.. index:: pair: class; Models::VarInit
.. _doxid-d8/dee/classModels_1_1VarInit:

class Models::VarInit
=====================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <models.h>
	
	class VarInit: public :ref:`Snippet::Init<doxid-d8/df6/classSnippet_1_1Init>`
	{
	public:
		// construction
	
		:target:`VarInit<doxid-d8/dee/classModels_1_1VarInit_1acb27fde6ac6eda66f81020561ca46a62>`(
			const :ref:`InitVarSnippet::Base<doxid-d3/d9e/classInitVarSnippet_1_1Base>`* snippet,
			const std::vector<double>& params
			);
	
		:target:`VarInit<doxid-d8/dee/classModels_1_1VarInit_1a07c4e76df4caa8a95ff5cf9cce6d7109>`(double constant);
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

