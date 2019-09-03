.. index:: pair: class; CodeGenerator::StructNameConstIter
.. _doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter:

template class CodeGenerator::StructNameConstIter
=================================================

.. toctree::
	:hidden:

Custom iterator for iterating through the containers of structs with 'name' members.


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <codeGenUtils.h>
	
	template <typename BaseIter>
	class StructNameConstIter: public BaseIter
	{
	public:
		// methods
	
		:target:`StructNameConstIter<doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter_1a5fcc38ded12de6940031d7e2157dfe11>`();
		:target:`StructNameConstIter<doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter_1a9a947634667b7f05af4d683a88c45dcb>`(BaseIter iter);
		const std::string* :target:`operator -><doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter_1a80b19896de3838c7a5f4ecd3105f816f>` () const;
		const std::string& :target:`operator *<doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter_1a48ade025dbe816e508063113c3c0ecca>` () const;
	};
