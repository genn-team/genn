.. index:: pair: namespace; Snippet
.. _doxid-df/daa/namespaceSnippet:

namespace Snippet
=================

.. toctree::
	:hidden:

	class_Snippet_Base.rst
	class_Snippet_Init.rst
	class_Snippet_ValueBase.rst
	class_Snippet_ValueBase-2.rst

Wrapper to ensure at compile time that correct number of values are used when specifying the values of a model's parameters and initial state.


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace Snippet {

	// classes

	class :ref:`Base<doxid-db/d97/classSnippet_1_1Base>`;

	template <typename SnippetBase>
	class :ref:`Init<doxid-d8/df6/classSnippet_1_1Init>`;

	template <>
	class :ref:`ValueBase<0><doxid-dd/df2/classSnippet_1_1ValueBase_3_010_01_4>`;

	template <size_t NumVars>
	class :ref:`ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`;

	} // namespace Snippet
