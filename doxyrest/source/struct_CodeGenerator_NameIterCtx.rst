.. index:: pair: struct; CodeGenerator::NameIterCtx
.. _doxid-df/d6f/structCodeGenerator_1_1NameIterCtx:

template struct CodeGenerator::NameIterCtx
==========================================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <codeGenUtils.h>
	
	template <typename Container>
	struct NameIterCtx
	{
		// typedefs
	
		typedef :ref:`StructNameConstIter<doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter>`<typename Container::const_iterator> :target:`NameIter<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx_1ab7662095b95fa77a874a674a7a3f06fc>`;

		// fields
	
		const Container :target:`container<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx_1a7fb1b7953fa37ee52a8076519ec02b94>`;
		const :ref:`NameIter<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx_1ab7662095b95fa77a874a674a7a3f06fc>` :target:`nameBegin<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx_1a8950e0b4d0debd52243efeb6e069b509>`;
		const :ref:`NameIter<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx_1ab7662095b95fa77a874a674a7a3f06fc>` :target:`nameEnd<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx_1ab0cf2b7b77583444e29198c68a7ab4a9>`;

		// methods
	
		:target:`NameIterCtx<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx_1a48637bc260812a5af43e0857ec978478>`(const Container& c);
	};
