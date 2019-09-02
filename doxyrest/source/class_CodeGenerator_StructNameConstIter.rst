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
		// construction
	
		:target:`StructNameConstIter<doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter_1a5fcc38ded12de6940031d7e2157dfe11>`();
		:target:`StructNameConstIter<doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter_1a9a947634667b7f05af4d683a88c45dcb>`(BaseIter iter);

		// methods
	
		const std::string* :target:`operator -><doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter_1aa1e8d05be4d929d18f7386d9b50701a2>` () const;
		const std::string& :target:`operator *<doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter_1aca25efd5bd86d7a1df530bd32a70a2f1>` () const;
	};
